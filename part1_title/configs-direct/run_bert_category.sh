python save_dataloader.py --yaml_config ./configs-direct/bow_content_category_select/BERT_category/BERT_Category_save_dataloader.yaml
python main.py --yaml_config ./configs-direct/bow_content_category_select/BERT_category/BERT_Category-train.yaml --method bow_content
python main.py --yaml_config ./configs-direct/bow_content_category_select/BERT_category/BERT_Category-test.yaml --method bow_content

python save_dataloader.py --yaml_config ./configs-direct/bow_title_category_select/BERT_category/BERT_Category_save_dataloader.yaml
python main.py --yaml_config ./configs-direct/bow_title_category_select/BERT_category/BERT_Category-train.yaml --method bow_title
python main.py --yaml_config ./configs-direct/bow_title_category_select/BERT_category/BERT_Category-test.yaml --method bow_title

python save_dataloader.py --yaml_config ./configs-direct/ngram_content_category_select/BERT_category/BERT_Category_save_dataloader.yaml
python main.py --yaml_config ./configs-direct/ngram_content_category_select/BERT_category/BERT_Category-train.yaml --method ngram_content
python main.py --yaml_config ./configs-direct/ngram_content_category_select/BERT_category/BERT_Category-test.yaml --method ngram_content

python save_dataloader.py --yaml_config ./configs-direct/ngram_title_category_select/BERT_category/BERT_Category_save_dataloader.yaml
python main.py --yaml_config ./configs-direct/ngram_title_category_select/BERT_category/BERT_Category-train.yaml --method ngram_title
python main.py --yaml_config ./configs-direct/ngram_title_category_select/BERT_category/BERT_Category-test.yaml --method ngram_title

python save_dataloader.py --yaml_config ./configs-direct/sentence_embedding_content_category_select/BERT_category/BERT_Category_save_dataloader.yaml
python main.py --yaml_config ./configs-direct/sentence_embedding_content_category_select/BERT_category/BERT_Category-train.yaml --method sentence_embedding_content
python main.py --yaml_config ./configs-direct/sentence_embedding_content_category_select/BERT_category/BERT_Category-test.yaml --method sentence_embedding_content

python save_dataloader.py --yaml_config ./configs-direct/sentence_embedding_title_category_select/BERT_category/BERT_Category_save_dataloader.yaml
python main.py --yaml_config ./configs-direct/sentence_embedding_title_category_select/BERT_category/BERT_Category-train.yaml --method sentence_embedding_title
python main.py --yaml_config ./configs-direct/sentence_embedding_title_category_select/BERT_category/BERT_Category-test.yaml --method sentence_embedding_title

python save_dataloader.py --yaml_config ./configs-direct/tfidf_content_category_select/BERT_category/BERT_Category_save_dataloader.yaml
python main.py --yaml_config ./configs-direct/tfidf_content_category_select/BERT_category/BERT_Category-train.yaml --method tfidf_content
python main.py --yaml_config ./configs-direct/tfidf_content_category_select/BERT_category/BERT_Category-test.yaml --method tfidf_content

python save_dataloader.py --yaml_config ./configs-direct/tfidf_title_category_select/BERT_category/BERT_Category_save_dataloader.yaml
python main.py --yaml_config ./configs-direct/tfidf_title_category_select/BERT_category/BERT_Category-train.yaml --method tfidf_title
python main.py --yaml_config ./configs-direct/tfidf_title_category_select/BERT_category/BERT_Category-test.yaml --method tfidf_title