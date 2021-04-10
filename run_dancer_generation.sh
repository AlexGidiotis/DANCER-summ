python src/dancer_generation.py \
    --mode dancer \
    --model_path dancer_pubmed/models \
    --data_path /home/jupyter/pubmed-dataset/processed/pubmed/test.json \
    --text_column document \
    --summary_column summary \
    --seed 100 \
    --test_batch_size 6 \
    --max_source_length 512 --max_summary_length 128 \
    --num_beams 5
