python3 main.py \
    --do_predict \
    --model_name_or_path=XinXuNLPer/MuseCoco_text2attribute \
    --test_file=/Users/jkimstylez/Code/google_colab/musecoco/1-text2attribute_model/data/predict.json \
    --attributes=data/att_key.json \
    --num_labels=num_labels.json \
    --output_dir=./tmp \
    --overwrite_output_dir
