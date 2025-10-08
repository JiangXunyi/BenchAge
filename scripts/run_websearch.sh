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
  python -m src.runners.websearch "$ds" \
         --model gpt-4o-mini &
done

wait 
