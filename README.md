# Myers-Briggs Personality Classification and Personality-Specific Language Generation Using Pretrained Language Model

Make sure you have pytorch installed, then do
```
pip install pytorch-pretrained-bert
```

First run the scraper files to scrape data, then run 
```
python classification/dataset_combiner
```

## Classification:

Run classification model by calling:
``` 
python classification/personality_classifier_main.py --do_train --do_eval
```

## Language Generation:
Download pre-trained weights from Google BERT-Base, Cased model (https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip) and save to language_generation/data/weights 

Convert tensorflow to pytorch-compatible weights by calling:
```
export BERT_BASE_DIR=language_generation/data/weights/cased_L-12_H-768_A-12

python language_generation/convert_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path $BERT_BASE_DIR/bert_model.ckpt \
    --bert_config_file $BERT_BASE_DIR/bert_config.json \
    --pytorch_dump_path $BERT_BASE_DIR/pytorch_model.bin
```

Create training data by running:
```
python language_generation/create_training_data.py
```


Run language generation by calling
```
python language_generation/personality_language_generation.py \
    --data_dir=language_generation/data/input/<type>
    --output_dir=language_generation/outputs/<type>
```

For example to run language generation for type ENFJ, run:
```
python language_generation/personality_language_generation.py \
--data_dir=language_generation/data/input/ENFJ
--output_dir=language_generation/outputs/ENFJ
```

