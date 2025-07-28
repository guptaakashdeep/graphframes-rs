use crate::pregel::PREGEL_MSG;
use crate::{GraphFrame, VERTEX_ID};
use datafusion::arrow::array::{
    Array, ArrayRef, Int32Array, Int32Builder, Int64Array, Int64Builder, MapBuilder,
};
use datafusion::arrow::datatypes::{DataType, Field, Fields};
use datafusion::common::ScalarValue;
use datafusion::error::{DataFusionError, Result};
use datafusion::functions::core::expr_ext::FieldAccessor;
use datafusion::functions_nested::map::map;
use datafusion::logical_expr::Volatility;
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
                let values = value.values().as_any().downcast_ref::<Int32Array>().unwrap();
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
        let mut map_builder = MapBuilder::new(None, l_marks_builder, distances_builder);

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

impl GraphFrame {
    /// Compute the shortest paths from landmarks to all vertices in the graph.
    ///
    /// # Arguments
    ///
    /// * `landmarks` - A vector of vertex IDs to use as landmarks
    /// * `max_iterations` - Optional maximum number of iterations to run
    ///
    /// # Returns
    ///
    /// A DataFrame with vertex IDs and a map of landmark -> distance
    pub async fn shortest_paths(
        &self,
        landmarks: Vec<i64>,
        max_iterations: Option<usize>,
    ) -> Result<DataFrame> {
        let internal_distances_data_type = DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(Fields::from(vec![
                    Field::new("key", DataType::Int64, false),
                    Field::new("value", DataType::Int32, false),
                ])),
                false,
            )),
            false,
        );
        const PARTICIPATING: &str = "participating";
        let landmarks_list = landmarks
            .iter()
            .map(|lm| lit(lm.clone()))
            .collect::<Vec<_>>();
        let zero_map = map(landmarks_list.clone(), vec![lit(i32::MAX); landmarks.len()]);

        // For landmarks:
        // - distance to itself is 0
        // - distance to other landmarks is infinity
        // For non-landmarks:
        // - distance to other landmarks is infinity
        let init_distances = when(
            col(VERTEX_ID).in_list(landmarks_list.clone(), true),
            zero_map,
        )
        .otherwise(map(
            landmarks_list.clone(),
            landmarks
                .iter()
                .map(|lm| {
                    when(col(VERTEX_ID).eq(lit(lm.clone())), lit(0))
                        .otherwise(lit(i32::MAX))
                        .unwrap()
                })
                .collect::<Vec<_>>(),
        ))?;

        let update_distances = map(
            landmarks_list,
            landmarks
                .iter()
                .map(|lm| {
                    let pregel_el = map_extract(col(PREGEL_MSG), lit(lm.clone()));
                    let map_el = map_extract(col(PREGEL_MSG), lit(lm.clone()));
                    when(
                        pregel_el
                            .clone()
                            .is_null()
                            .or(pregel_el.clone().lt_eq(map_el.clone())),
                        map_el,
                    )
                    .otherwise(pregel_el)
                    .unwrap()
                })
                .collect::<Vec<_>>(),
        );

        let landmarks_copy = Arc::new(landmarks.clone());

        let aggregate_expr = create_udaf(
            "merge_distance_maps",
            vec![internal_distances_data_type.clone()],
            Arc::new(internal_distances_data_type.clone()),
            Volatility::Immutable,
            Arc::new(move |_| Ok(Box::new(DistancesMap::new(landmarks_copy.clone())))),
            Arc::new(vec![internal_distances_data_type.clone()]),
        );

        // Initialize participation: only landmarks participate initially
        let init_participating = landmarks.iter().fold(lit(false), |acc, &landmark| {
            acc.or(col(VERTEX_ID).eq(lit(landmark)))
        });

        // Update participation condition: a vertex participates if it already participates or
        // if it receives a message (meaning it's a neighbor of a participating vertex)
        let update_participating = col(PARTICIPATING).or(col("msg").is_not_null());

        // Message to send: current distances map
        let message_expr = col(DISTANCES);

        // Aggregate expression: for each landmark, take the minimum of current distance and received distance + 1
        let aggregate_expr = named_struct(
            landmarks
                .iter()
                .flat_map(|&landmark| {
                    let landmark_str = landmark.to_string();
                    let current_dist = col(DISTANCES).field(&landmark_str);
                    let msg_dist = col("msg").field(&landmark_str);

                    // If message contains a distance for this landmark, consider it + 1 as a candidate
                    // Choose the minimum of current distance and received distance + 1
                    // If both are NULL, result is NULL
                    vec![
                        lit(landmark_str),
                        when(
                            msg_dist.clone().is_not_null(),
                            when(
                                current_dist.clone().is_not_null(),
                                least(vec![current_dist.clone(), msg_dist.clone() + lit(1)]),
                            )
                            .otherwise(msg_dist.clone() + lit(1))
                            .expect("Failed to create inner when expression"),
                        )
                        .otherwise(current_dist.clone())
                        .expect("Failed to create outer when expression"),
                    ]
                })
                .collect::<Vec<_>>(),
        );

        // Voting condition: vertex votes to halt if its distances map didn't change
        let voting_condition = col(DISTANCES).eq(col("updated_distances"));

        // Run Pregel algorithm
        let result = self
            .pregel()
            // Add vertex columns
            .add_vertex_column(DISTANCES, init_distances.clone(), col("updated_distances"))
            .add_vertex_column("updated_distances", init_distances, aggregate_expr.clone())
            // Set participation condition
            .with_participation_column(PARTICIPATING, init_participating, update_participating)
            // Add message
            .add_message(message_expr, crate::pregel::MessageDirection::SrcToDst)
            // Set aggregate expression
            .with_aggregate_expr(aggregate_expr)
            // Set voting condition
            .with_vertex_voting("active", voting_condition)
            // Set max iterations if provided
            .max_iterations(max_iterations.unwrap_or(usize::MAX))
            // Run the algorithm
            .run(false)
            .await?;

        // Return the result with vertex ID and distances map
        Ok(result.data.select(vec![col(VERTEX_ID), col(DISTANCES)])?)
    }
}
