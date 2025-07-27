use crate::pregel::{pregel_src, MessageDirection, PregelBuilder, PREGEL_MSG};
use crate::{GraphFrame, VERTEX_ID};
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
    include_debug_columns: bool
}

impl<'a> PageRank<'a> {

    pub fn new(graph: &'a GraphFrame) -> Self {
        PageRank {
            graph,
            max_iter: 0,
            reset_prob: 0.15,
            include_debug_columns: false
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
                            // .with_column_renamed(VERTEX_ID, &format!("{}_right", VERTEX_ID))?;

        // Get the vertices with the degrees to send the messages
        // let vertices_with_degrees = self.graph.vertices.clone().join(
        //     out_degree_df,
        //     JoinType::Left,
        //     &[VERTEX_ID],
        //     &[&format!("{}_right", VERTEX_ID)],
        //     None
        // )?
        // .drop_columns(&[&format!("{}_right", VERTEX_ID)])?
        // .with_column("out_degree", coalesce(vec![col("out_degree"), lit(0)]))?;

        // Create a temp graph that has vertices with degrees that will be used in Pregel Execution
        let graph_with_degrees = GraphFrame{
            vertices: vertices_with_degrees,
            edges: self.graph.edges.clone()
        };


        // --- Step 2: Create Pregel builder ---
        let pregel_builder = PregelBuilder::new(graph_with_degrees)
            .max_iterations(self.max_iter)
            .add_vertex_column("pagerank",
                lit(1.0 / num_vertices), // All vertices start with a rank of 1/N
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
        let final_page_ranks = result.data;
        
        // TODO: Do we want to return the full computed dataframe or just vertex_id and page_rank should be fine??
        Ok(final_page_ranks.select(vec![col("id"), col("pagerank")])?)
        // Ok(final_page_ranks)
    }
}