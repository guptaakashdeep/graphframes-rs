mod pregel;
pub mod pagerank;

use datafusion::error::Result;
use datafusion::functions_aggregate::count::count;
use datafusion::prelude::*;

use crate::pagerank::PageRank;

pub const VERTEX_ID: &str = "id";
pub const EDGE_SRC: &str = "src";
pub const EDGE_DST: &str = "dst";

#[derive(Debug, Clone)]
pub struct GraphFrame {
    vertices: DataFrame,
    edges: DataFrame,
}

impl GraphFrame {
    pub async fn num_nodes(&self) -> Result<i64> {
        let count = self.vertices.clone().count().await?;
        Ok(count as i64)
    }

    pub async fn num_edges(&self) -> Result<i64> {
        let count = self.edges.clone().count().await?;
        Ok(count as i64)
    }

    pub async fn in_degrees(&self) -> Result<DataFrame> {
        let df = self.edges.clone().aggregate(
            vec![col(EDGE_DST)],
            vec![count(col(EDGE_SRC)).alias("in_degree")],
        )?;
        Ok(df.select(vec![col(EDGE_DST).alias(VERTEX_ID), col("in_degree")])?)
    }

    pub async fn out_degrees(&self) -> Result<DataFrame> {
        let df = self.edges.clone().aggregate(
            vec![col(EDGE_SRC)],
            vec![count(col(EDGE_DST)).alias("out_degree")],
        )?;
        Ok(df.select(vec![col(EDGE_SRC).alias(VERTEX_ID), col("out_degree")])?)
    }

    pub fn pagerank(&self) -> PageRank {
        PageRank::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{Int64Array, RecordBatch, StringArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use std::collections::HashMap;
    use std::sync::Arc;

    fn create_test_graph() -> Result<GraphFrame> {
        let ctx = SessionContext::new();

        let vertices_data = RecordBatch::try_new(
            SchemaRef::from(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("name", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10])),
                Arc::new(StringArray::from(vec![
                    "Hub", "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry",
                    "Ivy",
                ])),
            ],
        );
        let vertices = ctx.read_batch(vertices_data?)?;

        let edges_data = RecordBatch::try_new(
            SchemaRef::from(Schema::new(vec![
                Field::new("src", DataType::Int64, false),
                Field::new("dst", DataType::Int64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7,
                    8, 8, 9, 10,
                ])),
                Arc::new(Int64Array::from(vec![
                    2, 3, 4, 5, 6, 7, 8, 9, 10, 3, 4, 5, 6, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 8, 9,
                    9, 10, 10, 1,
                ])),
            ],
        );
        let edges = ctx.read_batch(edges_data?)?;

        Ok(GraphFrame { vertices, edges })
    }

    // Creates GraphFrame from ldbc datasets.
    // dataset: name of ldbc dataset like tested-pr-directed, wiki-Talk, etc.
    async fn create_ldbc_test_graph(dataset: &str) -> Result<GraphFrame>{
        let ctx = SessionContext::new();

        let manifest_dir = env!("CARGO_MANIFEST_DIR");

        let edge_schema = Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false)
        ]);
        let vertices_schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false)
        ]);

        let edges_path = format!("{}/testing/data/ldbc/{}/{}.e.csv", manifest_dir, dataset, dataset);
        let vertices_path = format!("{}/testing/data/ldbc/{}/{}.v.csv", manifest_dir, dataset, dataset);
        
        let edges = ctx.read_csv(
            &edges_path,
            CsvReadOptions::new()
                .delimiter(b' ')
                .has_header(false)
                .schema(&edge_schema)
        ).await?;

        let vertices = ctx.read_csv(
            &vertices_path,
            CsvReadOptions::new()
                .delimiter(b' ')
                .has_header(false)
                .schema(&vertices_schema)
        ).await?;

        Ok(GraphFrame{ vertices, edges})
    }

    // Gets the expected pagerank results from the mentioned ldbc dataset
    async fn get_ldbc_pr_results(dataset: &str) -> Result<DataFrame> {
        let ctx = SessionContext::new();
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let expected_pr_schema = Schema::new(vec![
            Field::new("vertex_id", DataType::Int64, false),
            Field::new("expected_pr", DataType::Float64, false)
        ]);
        let expected_pr_path = format!("{}/testing/data/ldbc/{}/{}-PR.csv", manifest_dir, dataset, dataset);
        let expected_pr = ctx.read_csv(
            &expected_pr_path,
            CsvReadOptions::new()
                .delimiter(b' ')
                .has_header(false)
                .schema(&expected_pr_schema)
        ).await?;
        Ok(expected_pr)
    }

    #[tokio::test]
    async fn test_num_nodes() -> Result<()> {
        let graph = create_test_graph()?;
        let num_nodes = graph.num_nodes().await?;
        assert_eq!(num_nodes, 10);
        Ok(())
    }

    #[tokio::test]
    async fn test_num_edges() -> Result<()> {
        let graph = create_test_graph()?;
        let num_edges = graph.num_edges().await?;
        assert_eq!(num_edges, 30);
        Ok(())
    }

    #[tokio::test]
    async fn test_in_degrees() -> Result<()> {
        let graph = create_test_graph()?;
        let df = graph.in_degrees().await?;
        let batches = df.collect().await?;

        let mut degree_map = HashMap::new();
        for batch in batches.iter() {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let degrees = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();

            for i in 0..ids.len() {
                degree_map.insert(ids.value(i), degrees.value(i));
            }
        }

        assert_eq!(degree_map.get(&1), Some(&1));
        assert_eq!(degree_map.get(&2), Some(&1));
        assert_eq!(degree_map.get(&3), Some(&2));
        assert_eq!(degree_map.get(&4), Some(&3));
        assert_eq!(degree_map.get(&5), Some(&4));
        assert_eq!(degree_map.get(&6), Some(&5));
        assert_eq!(degree_map.get(&7), Some(&4));
        assert_eq!(degree_map.get(&8), Some(&4));
        assert_eq!(degree_map.get(&9), Some(&3));
        assert_eq!(degree_map.get(&10), Some(&3));

        Ok(())
    }

    #[tokio::test]
    async fn test_out_degrees() -> Result<()> {
        let graph = create_test_graph()?;
        let df = graph.out_degrees().await?;
        let batches = df.collect().await?;

        let mut degree_map = HashMap::new();
        for batch in batches.iter() {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let degrees = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();

            for i in 0..ids.len() {
                degree_map.insert(ids.value(i), degrees.value(i));
            }
        }

        assert_eq!(degree_map.get(&1), Some(&9));
        assert_eq!(degree_map.get(&2), Some(&4));
        assert_eq!(degree_map.get(&3), Some(&3));
        assert_eq!(degree_map.get(&4), Some(&3));
        assert_eq!(degree_map.get(&5), Some(&3));
        assert_eq!(degree_map.get(&6), Some(&2));
        assert_eq!(degree_map.get(&7), Some(&2));
        assert_eq!(degree_map.get(&8), Some(&2));
        assert_eq!(degree_map.get(&9), Some(&1));
        assert_eq!(degree_map.get(&10), Some(&1));

        Ok(())
    }

    #[tokio::test]
    async fn test_pagerank_run() -> Result<()> {
        let test_dataset: &str = "test-pr-directed";
        let graph = create_ldbc_test_graph(test_dataset).await?;

        let calculated_page_rank = graph.pagerank()
                                        .max_iter(14)
                                        .reset_prob(0.15)
                                        .run()
                                        .await?;
        let ldbc_page_rank = get_ldbc_pr_results(test_dataset).await?;

        let comparison_df = calculated_page_rank
                .join(ldbc_page_rank, JoinType::Left, &["id"], &["vertex_id"], None)?
                .with_column("difference", col("pagerank") - col("expected_pr"))?
                .filter(col("difference").gt(lit(0.0015)))?;

        // comparison_df.clone().show().await?;

        // Check if there are no pageranks with difference more than 0.0015
        assert_eq!(comparison_df.count().await?, 0);

        Ok(())
    }
}
