#!/usr/bin/env bash

datasets=(
  NaturalQuestion
  TruthfulQA
  BoolQ
  FreshQA
  HaluEval
  SelfAware
  TriviaQA
)

for ds in "${datasets[@]}"; do
  python -m src.evaluator.evaluator "$ds" \
  &
done

wait 
