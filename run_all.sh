#!/bin/bash
set -e

echo "═══════════════════════════════════════════════"
echo "  1/7: Pipeline (DuckDB → Parquet → CSV)"
echo "═══════════════════════════════════════════════"
python -m src.pipeline.run_pipeline

echo ""
echo "═══════════════════════════════════════════════"
echo "  2/7: Survival Analysis (Cox PH, real data)"
echo "═══════════════════════════════════════════════"
python -m src.survival.churn

echo ""
echo "═══════════════════════════════════════════════"
echo "  3/7: Segmentation (K-Means)"
echo "═══════════════════════════════════════════════"
python -m src.segmentation.cluster

echo ""
echo "═══════════════════════════════════════════════"
echo "  4/7: Observational Study (PSM)"
echo "═══════════════════════════════════════════════"
python -m src.ab_testing.observational

echo ""
echo "═══════════════════════════════════════════════"
echo "  5/7: RAG Evaluation"
echo "═══════════════════════════════════════════════"
python -m src.rag.run_eval

echo ""
echo "═══════════════════════════════════════════════"
echo "  6/7: RAG → Churn Simulation (n=500 + n=5000)"
echo "═══════════════════════════════════════════════"
python -m src.survival.rag_to_churn

echo ""
echo "═══════════════════════════════════════════════"
echo "  7/7: Finalize Outputs (all plots + dashboard)"
echo "═══════════════════════════════════════════════"
python -m src.pipeline.finalize_outputs

echo ""
echo "All done. Outputs in ./output/"
