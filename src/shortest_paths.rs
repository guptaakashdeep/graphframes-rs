use crate::pregel::{PREGEL_MSG, pregel_src};
use crate::{GraphFrame, VERTEX_ID};
use datafusion::arrow::array::{
    Array, ArrayRef, Int32Array, Int32Builder, Int64Array, Int64Builder, MapBuilder, MapFieldNames,
    as_map_array,
};
use datafusion::arrow::datatypes::{DataType, Field, Fields};
use datafusion::common::ScalarValue;
use datafusion::common::cast::{as_int32_array, as_int64_array};
use datafusion::error::{DataFusionError, Result};
use datafusion::functions_nested::map::map;
use datafusion::logical_expr::{ColumnarValue, Volatility};
use datafusion::physical_plan::Accumulator;
use datafusion::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Column name for distances in the Shortest Paths algorithm
pub const DISTANCES: &str = "distances";

#[derive(Debug)]
struct DistancesMap {
    distances: HashMap<i64, i32>,
}

impl DistancesMap {
    fn new(landmarks: Arc<Vec<i64>>) -> Self {
        Self {
            distances: HashMap::from_iter(landmarks.iter().map(|&lm| (lm, i32::MAX))),
        }
    }
}

impl Accumulator for DistancesMap {
    // Internal state: Map<i64, i32> -- mapping from landmarks to distances;
    // Update logic is the same as merge: for all the keys taking the minimal value;
    // Result is the same as state and input rows;
    //
    // Transform multiple Map<i64, i32> into a single Map<i64, i32> with minimal values per key.
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        let array_of_maps = &values[0];
        (0..array_of_maps.len()).try_for_each(|index| {
            let new_map = ScalarValue::try_from_array(array_of_maps, index)?;

            if let ScalarValue::Map(value) = new_map {
                let keys = value.keys().as_any().downcast_ref::<Int64Array>().unwrap();
                let values = value
                    .values()
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();
                for (k, v) in keys.iter().zip(values.iter()) {
                    match (k, v) {
                        (Some(k), Some(v)) => {
                            if v < self.distances[&k] {
                                self.distances.insert(k, v);
                            }
                        }
                        _ => {
                            return Err(DataFusionError::Plan(
                                "Invalid map entry in DistancesMap accumulator".to_string(),
                            ));
                        }
                    }
                }
            }
            Ok(())
        })
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        self.state().map(|state| state[0].clone())
    }

    fn size(&self) -> usize {
        size_of::<Self>()
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let l_marks_builder = Int64Builder::with_capacity(self.distances.len());
        let distances_builder = Int32Builder::with_capacity(self.distances.len());
        let mut map_builder = MapBuilder::new(
            Some(MapFieldNames {
                entry: "entries".to_string(),
                key: "key".to_string(),
                value: "value".to_string(),
            }),
            l_marks_builder,
            distances_builder,
        );

        map_builder.keys().append_array(&Int64Array::from(
            self.distances.keys().map(|k| k.clone()).collect::<Vec<_>>(),
        ));
        map_builder.values().append_array(&Int32Array::from(
            self.distances
                .values()
                .map(|v| v.clone())
                .collect::<Vec<_>>(),
        ));
        map_builder.append(true)?;
        Ok(vec![ScalarValue::Map(Arc::new(map_builder.finish()))])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.update_batch(states)
    }
}

fn merge_distances_maps(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let arrays = ColumnarValue::values_to_arrays(args)?;
    let left_map = as_map_array(&arrays[0]);
    let right_map = as_map_array(&arrays[1]);

    if left_map.len() != right_map.len() {
        return Err(DataFusionError::Plan(format!(
            "Invalid map length in merge_distances_maps: {} != {}",
            left_map.len(),
            right_map.len()
        )));
    }
    let keys = as_int64_array(left_map.keys())?;
    let left_values = as_int32_array(left_map.values())?;
    let right_values = as_int32_array(right_map.values())?;

    let new_keys_builder = Int64Builder::with_capacity(left_map.len());
    let new_values_builder = Int32Builder::with_capacity(left_map.len());
    let mut builder = MapBuilder::new(
        Some(MapFieldNames {
            entry: "entries".to_string(),
            key: "key".to_string(),
            value: "value".to_string(),
        }),
        new_keys_builder,
        new_values_builder,
    );

    for map_idx in 0..left_map.len() {
        if left_values.is_null(map_idx) || right_values.is_null(map_idx) {
            builder.append(false)?;
            continue;
        }
        let left_start = left_map.value_offsets()[map_idx] as usize;
        let left_end = left_map.value_offsets()[map_idx + 1] as usize;

        let right_start = right_map.value_offsets()[map_idx] as usize;

        for (pos, i) in (left_start..left_end).enumerate() {
            builder.keys().append_value(keys.value(i));
            let left_value = left_values.value(i);
            let right_value = right_values.value(right_start + pos);
            if left_value < right_value {
                builder.values().append_value(left_value);
            } else {
                builder.values().append_value(right_value);
            }
        }
        builder.append(true)?;
    }

    Ok(ColumnarValue::Array(Arc::new(builder.finish())))
}

pub struct ShortestPathsBuilder<'a> {
    graph_frame: &'a GraphFrame,
    landmarks: Vec<i64>,
    max_iterations: usize,
    checkpoint_interval: usize,
}

impl<'a> ShortestPathsBuilder<'a> {
    pub fn new(graph_frame: &'a GraphFrame, landmarks: Vec<i64>) -> Self {
        Self {
            graph_frame,
            landmarks,
            max_iterations: i32::MAX as usize,
            checkpoint_interval: 2,
        }
    }

    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }
    pub fn checkpoint_interval(mut self, checkpoint_interval: usize) -> Self {
        self.checkpoint_interval = checkpoint_interval;
        self
    }

    pub async fn run(self) -> Result<DataFrame> {
        let internal_distances_data_type = DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(Fields::from(vec![
                    Field::new("key", DataType::Int64, false),
                    Field::new("value", DataType::Int32, true),
                ])),
                false,
            )),
            false,
        );
        const PARTICIPATING: &str = "participating";
        let landmarks_list = self
            .landmarks
            .iter()
            .map(|lm| lit(lm.clone()))
            .collect::<Vec<_>>();
        let zero_map = map(
            landmarks_list.clone(),
            vec![lit(i32::MAX); self.landmarks.len()],
        );

        // For landmarks:
        // - distance to itself is 0
        // - distance to other landmarks is infinity
        // For non-landmarks:
        // - distance to other landmarks is infinity
        let init_distances = self
            .landmarks
            .clone()
            .iter()
            .fold(
                when(
                    col(VERTEX_ID).in_list(landmarks_list.clone(), true),
                    zero_map.clone(),
                ),
                |mut builder, &landmark| {
                    builder.when(
                        col(VERTEX_ID).eq(lit(landmark)),
                        map(
                            landmarks_list.clone(),
                            self.landmarks
                                .clone()
                                .iter()
                                .map(|lm_value| {
                                    if lm_value == &landmark {
                                        lit(0i32)
                                    } else {
                                        lit(i32::MAX)
                                    }
                                })
                                .collect::<Vec<_>>(),
                        ),
                    )
                },
            )
            .otherwise(zero_map.clone())?; // otherwise should be unreachable

        let update_distances = create_udf(
            "_merge_two_maps",
            vec![
                internal_distances_data_type.clone(),
                internal_distances_data_type.clone(),
            ],
            internal_distances_data_type.clone(),
            Volatility::Immutable,
            Arc::new(merge_distances_maps),
        );

        let landmarks_copy = Arc::new(self.landmarks.clone());

        let aggregate_expr_udaf = create_udaf(
            "_merge_distance_maps",
            vec![internal_distances_data_type.clone()],
            Arc::new(internal_distances_data_type.clone()),
            Volatility::Immutable,
            Arc::new(move |_| Ok(Box::new(DistancesMap::new(landmarks_copy.clone())))),
            Arc::new(vec![internal_distances_data_type.clone()]),
        );

        // Initialize participation: only landmarks participate initially
        let init_participating = self.landmarks.iter().fold(lit(false), |acc, &landmark| {
            acc.or(col(VERTEX_ID).eq(lit(landmark)))
        });

        // Update participation condition: a vertex participates if it already participates or
        // if it receives a message (meaning it's a neighbor of a participating vertex)
        let update_participating =
            col(PREGEL_MSG)
                .is_not_null()
                .and(map_values(col(DISTANCES)).not_eq(map_values(
                    update_distances.call(vec![col(DISTANCES), col(PREGEL_MSG)]),
                )));

        // Message to send: current distances map
        let message_expr = pregel_src(DISTANCES);

        // Run Pregel algorithm
        let result = self
            .graph_frame
            .pregel()
            // Add vertex columns
            .add_vertex_column(
                DISTANCES,
                init_distances.clone(),
                update_distances.call(vec![col(DISTANCES), col(PREGEL_MSG)]),
            )
            // Set participation condition
            .with_participation_column(
                PARTICIPATING,
                init_participating,
                update_participating.clone(),
            )
            // Add a message
            .add_message(message_expr, crate::pregel::MessageDirection::SrcToDst)
            // Set aggregate expression
            .with_aggregate_expr(aggregate_expr_udaf.call(vec![col(PREGEL_MSG)]))
            // Set voting condition
            .with_vertex_voting("active", update_participating)
            // Set max iterations if provided
            .max_iterations(self.max_iterations)
            // Set checkpoint interval
            .checkpoint_interval(2)
            // Run the algorithm
            .run(false)
            .await?;

        // Return the result with vertex ID and distances map
        Ok(result.data.select(vec![col(VERTEX_ID), col(DISTANCES)])?)
    }
}

impl GraphFrame {
    pub fn shortest_paths(&self, landmarks: Vec<i64>) -> ShortestPathsBuilder {
        ShortestPathsBuilder::new(self, landmarks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{Int32Array, Int64Array, RecordBatch};
    use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use datafusion::prelude::SessionContext;
    use std::sync::Arc;

    fn create_small_test_graph() -> Result<GraphFrame> {
        let ctx = SessionContext::new();

        let vertices_data = RecordBatch::try_new(
            SchemaRef::from(Schema::new(vec![Field::new("id", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3, 4]))],
        )?;
        let vertices = ctx.read_batch(vertices_data)?;

        let edges_data = RecordBatch::try_new(
            SchemaRef::from(Schema::new(vec![
                Field::new("src", DataType::Int64, false),
                Field::new("dst", DataType::Int64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 2, 3])),
                Arc::new(Int64Array::from(vec![2, 3, 4, 4])),
            ],
        )?;
        let edges = ctx.read_batch(edges_data)?;

        Ok(GraphFrame { vertices, edges })
    }

    #[tokio::test]
    async fn test_shortest_paths_single_landmark() -> Result<()> {
        let graph = create_small_test_graph()?;
        let landmarks = vec![1];
        let result = graph.shortest_paths(landmarks).run().await?;
        let batches = result.collect().await?;

        for batch in batches {
            let ids = batch
                .column_by_name(VERTEX_ID)
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();

            for i in 0..ids.len() {
                let id = ids.value(i);
                let distances = batch.column(1);
                match id {
                    1 => assert_eq!(map_extract_value(distances, i, 1), 0),
                    2 => assert_eq!(map_extract_value(distances, i, 1), 1),
                    3 => assert_eq!(map_extract_value(distances, i, 1), 2),
                    4 => assert_eq!(map_extract_value(distances, i, 1), 3),
                    _ => panic!("Unexpected vertex id"),
                }
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_shortest_paths_multiple_landmarks() -> Result<()> {
        let graph = create_small_test_graph()?;
        let landmarks = vec![1, 4];
        let result = graph.shortest_paths(landmarks).run().await?;
        let batches = result.collect().await?;

        for batch in batches {
            let ids = batch
                .column_by_name(VERTEX_ID)
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();

            for i in 0..ids.len() {
                let id = ids.value(i);
                let distances = batch.column(1);
                match id {
                    1 => {
                        assert_eq!(map_extract_value(distances, i, 1), 0);
                        assert_eq!(map_extract_value(distances, i, 4), 3);
                    }
                    2 => {
                        assert_eq!(map_extract_value(distances, i, 1), 1);
                        assert_eq!(map_extract_value(distances, i, 4), 2);
                    }
                    3 => {
                        assert_eq!(map_extract_value(distances, i, 1), 2);
                        assert_eq!(map_extract_value(distances, i, 4), 1);
                    }
                    4 => {
                        assert_eq!(map_extract_value(distances, i, 1), 3);
                        assert_eq!(map_extract_value(distances, i, 4), 0);
                    }
                    _ => panic!("Unexpected vertex id"),
                }
            }
        }
        Ok(())
    }

    fn map_extract_value(array: &ArrayRef, row: usize, key: i64) -> i32 {
        let scalar = ScalarValue::try_from_array(array, row).unwrap();
        if let ScalarValue::Map(map) = scalar {
            let keys = map.keys().as_any().downcast_ref::<Int64Array>().unwrap();
            let values = map.values().as_any().downcast_ref::<Int32Array>().unwrap();
            for (i, k) in keys.iter().enumerate() {
                if k == Some(key) {
                    return values.value(i);
                }
            }
        }
        panic!("Key not found in map")
    }
}
