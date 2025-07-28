use crate::pregel::{pregel_src, MessageDirection, PregelBuilder, PREGEL_MSG};
use crate::GraphFrame;
use datafusion::error::Result;
use datafusion::functions_aggregate::sum::sum;
use datafusion::prelude::*;

/// A builder for the PageRank algorithm.
/// 
#[derive(Debug)]
pub struct PageRank<'a> {
    graph: &'a GraphFrame,
    max_iter: usize,
    reset_prob: f64,
    include_debug_columns: bool,
    checkpoint_interval: usize
}

impl<'a> PageRank<'a> {

    pub fn new(graph: &'a GraphFrame) -> Self {
        PageRank {
            graph,
            max_iter: 0,
            reset_prob: 0.15,
            include_debug_columns: false,
            checkpoint_interval: 2 // Revisit once default is finalized. temp for now.
        }
    }

    pub fn max_iter(mut self, iter: usize) -> Self {
        self.max_iter = iter;
        self
    }

    pub fn reset_prob(mut self, prob: f64) -> Self {
        self.reset_prob = prob;
        self
    }

    pub fn checkpoint_interval(mut self, checkpoint_interval: usize) -> Self {
        self.checkpoint_interval = checkpoint_interval;
        self
    }

    /// Execute PageRank algorithm.
    /// Execution via generic Pregel engine
    pub async fn run(&self) -> Result<DataFrame> {
        let num_vertices = self.graph.num_nodes().await? as f64;
        // Defaults to 0.85 if default reset_prob is used: 0.15
        let alpha = 1.0 - self.reset_prob;
        let reset_prob_per_vertices = self.reset_prob / num_vertices;

        // --- Step 1: Pre-calculate out-degrees ---
        // PageRank needs the out-degree of each vertex to distribute its rank.
        //contains vertex_id and put_degree for each vertices
        let vertices_with_degrees = self.graph.out_degrees().await?;

        // Create a temp graph that has vertices with degrees that will be used in Pregel Execution
        let graph_with_degrees = GraphFrame{
            vertices: vertices_with_degrees,
            edges: self.graph.edges.clone()
        };


        // --- Step 2: Create Pregel builder ---
        let pregel_builder = PregelBuilder::new(graph_with_degrees)
            .max_iterations(self.max_iter)
            .checkpoint_interval(self.checkpoint_interval)
            .add_vertex_column("pagerank",
                lit(reset_prob_per_vertices), // All vertices start with a rank of 1/N
                lit(reset_prob_per_vertices) + lit(alpha) * col(PREGEL_MSG) // PageRank calculation
            )
            .add_vertex_column(
                "out_degree",
                col("out_degree"),
                col("out_degree")
            ) // out_degrees are static
            .add_message(
                pregel_src("pagerank") / pregel_src("out_degree"),
                MessageDirection::SrcToDst
            )
            .with_aggregate_expr(sum(col(PREGEL_MSG)));

        // --- Step 3: Run Pregel Engine ---
        let result = pregel_builder.run(self.include_debug_columns).await?;
        let calculated_page_ranks = result.data;
        
        // Calculating the pagerank aggregation
        let aggregated_rank = calculated_page_ranks.clone().aggregate(
            vec![],
            vec![sum(col("pagerank")).alias("pagerank_sum")]
        )?
        .cache().await?;

        // Cross join with calculated pageranks df
        let final_page_ranks = calculated_page_ranks.join(
                    aggregated_rank,
                    JoinType::Inner,
                    &[],
                    &[],
                    None
                )?;
        Ok(final_page_ranks.select(
            vec![
                col("id"), 
                (col("pagerank") / col("pagerank_sum")).alias("pagerank")]
            )?)
    }
}
