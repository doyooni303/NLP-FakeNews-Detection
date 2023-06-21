methods=( bow_content bow_title ngram_content ngram_title sentence_embedding_content sentence_embedding_title tfidf_content tfidf_title random )
modelnames=( BERT_LSTM BERT_LSTM_double_layer BERT_LSTM_m2o DualBERT RoBERTa_Category RoBERTa_LSTM RoBERTa_DualBERT BERT_Sims_Stop_Gradient BERT_Weighted_Sims_Stop_Gradient Multimodal_net)

# Test
for method in ${methods[@]}
do
    for modelname in ${modelnames[@]}
    do
        python main.py \
        --yaml_config ./configs-direct/${method}_category_select/${modelname[@]}/${modelname[@]}-test.yaml \
        --method ${method}
    done
done
