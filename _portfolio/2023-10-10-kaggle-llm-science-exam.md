---
title: "Kaggle - LLM Science Exam"
excerpt: "Use LLMs to answer difficult science questions<br/><img src='/images/kaggle-llm-science-exam.png'>"
collection: portfolio
portfoliolink: "https://devpost.com/software/destchat](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446303#2477056"
date: 2023-10-10
---

# Introduction

First of all, we would like to express our sincere appreciation to the competition organizers and competitors who shared their valuable thoughts and resources during the competition.

Before starting, feel free to take a quick look at our **[notebook](https://www.kaggle.com/code/hqfang/kaggle-llmse-inference)** solution.  I would also like to introduce our team members: **[@yuekaixueirc](https://www.kaggle.com/yuekaixueirc)**, **[@lindseywei](https://www.kaggle.com/lindseywei)**, and **[@hqfang](https://www.kaggle.com/hqfang)**.

Our team's solution started from [@mbanaei](https://www.kaggle.com/mbanaei)'s [notebook](https://www.kaggle.com/code/mbanaei/86-2-with-only-270k-articles). We made changes in three different parts: **Context Retrieval**, **Model Inference**, and **Ensemble Models**.

<br>

# Context Retrieval

We kept the original RAG method in the notebook. On top of that, we borrowed one more context source from [@cdeotte](https://www.kaggle.com/cdeotte)'s [notebook](https://www.kaggle.com/code/cdeotte/how-to-train-open-book-model-part-2). Moreover, when retrieving relevant contexts using prompt and options, we found that [@mbanaei](https://www.kaggle.com/mbanaei)'s way of weighting the prompt by repeating it three times worked better, so we did it the same way for the context retrieval we added, that is:

    trn['answer_all'] = trn.apply(lambda x: " ".join([x['A'], x['B'], x['C'], x['D'], x['E']]), axis=1)
    trn['prompt_answer_stem'] = trn['prompt'] + " " + trn['prompt'] + " " + trn['prompt'] + " " + trn['answer_all']

We also found that [@mbanaei](https://www.kaggle.com/mbanaei)'s way of reversing the order of relevant context to make relevant contexts closer to the prompt and options also useful, so we did it the same way for the context retrieval we added, that is:

    contexts = []

    for r in tqdm(trn.itertuples(), total=len(trn)):

        prompt_id = r.Index

        prompt_indices =     processed_wiki_text_data[processed_wiki_text_data['document_id'].isin(wikipedia_file_data[wikipedia_file_data['prompt_id']==prompt_id]['id'].values)].index.values

        if prompt_indices.shape[0] > 0:
            prompt_index = faiss.index_factory(wiki_data_embeddings.shape[1], "Flat")
            prompt_index.add(wiki_data_embeddings[prompt_indices])

            context = ""
            context_temp = []
        
            ss, ii = prompt_index.search(question_embeddings, NUM_SENTENCES_INCLUDE)
            for _s, _i in zip(ss[prompt_id], ii[prompt_id]):
                context_temp.append(processed_wiki_text_data.loc[prompt_indices]['text'].iloc[_i])
            
            context_temp.reverse()
        
            for i in range(len(context_temp)):
                context += context_temp[i] + "\n"
            
        contexts.append(context)
    
    contexts_wiki = contexts
    del contexts
    gc.collect()

We also changed the variable `NUM_SENTENCES_INCLUDE` in the added retrieval to 15 in order to make our contexts have less irrelevant information.

Also, Inspired by [@simjeg](https://www.kaggle.com/simjeg)'s [notebook](https://www.kaggle.com/code/simjeg/platypus2-70b-with-wikipedia-rag), we made use of the variable `IS_TEST_SET` to save local run time for context retrieval.

# Model Inference

Instead of using a Longformer, we chose to use DeBERTa as it outperformed the Longformer in our experiments. Considering to add more diversity to the inference, also inspired by [@itsuki9180](https://www.kaggle.com/itsuki9180)'s [notebook](https://www.kaggle.com/code/itsuki9180/llm-sciex-optimise-ensemble-weights), we decided to use both OpenBook models and non-OpenBook models.

Among the OpenBook models, we used three DeBERTas trained locally by [@yuekaixueirc](https://www.kaggle.com/yuekaixueirc) primarily using [@cdeotte](https://www.kaggle.com/cdeotte)'s [notebook](https://www.kaggle.com/code/cdeotte/how-to-train-open-book-model-part-1). Among the non-OpenBook models, we used one DeBERTa trained locally by [@hqfang](https://www.kaggle.com/hqfang) primarily using [@radek1](https://www.kaggle.com/radek1)'s [notebook](https://www.kaggle.com/code/radek1/new-dataset-deberta-v3-large-training), one DeBERTa trained locally by [@lindseywei](https://www.kaggle.com/lindseywei) using the [LoRA](https://www.kaggle.com/code/datafan07/single-model-rewardtrainer-lora-llm/notebook) technique, and one DeBERTa posted publicly by [@itsuki9180](https://www.kaggle.com/itsuki9180) using the [AWP](https://www.kaggle.com/code/itsuki9180/introducing-adversarial-weight-perturbation-awp) technique.

All of the models we trained are posted publicly in this **[dataset](https://www.kaggle.com/datasets/hqfang/kaggle-llmse-dataset)**.

For the inference part of the OpenBook models, we changed the way of tokenizing samples to the one we used for training, that is:

    def prepare_answering_input(
            tokenizer, 
            question,  
            options,   
            context,   
            max_seq_length=1024,
        ):
    
        first_sentence = [ "[CLS] " + context ] * 5
        second_sentences = [" #### " + question + " [SEP] " + options[option] + " [SEP]" for option in range(0,5)]

        tokenized_examples = tokenizer(
            first_sentence, second_sentences,
            max_length=max_seq_length,
            padding="longest",
            truncation="only_first",
            return_tensors="pt",
            add_special_tokens=False
        )
    
        input_ids = tokenized_examples['input_ids'].unsqueeze(0)
        attention_mask = tokenized_examples['attention_mask'].unsqueeze(0)
        example_encoded = {
            "input_ids": input_ids.to(model.device.index),
            "attention_mask": attention_mask.to(model.device.index),
        }
    
        return example_encoded

Note that we also changed the `max_seq_length` to 1024.

We kept the inference code as it was, but changed some details to make it only output the probabilities for further use. What's more, we assigned weights of the three predictions of different contexts as 4:4:2. See more below:

    def get_predictions_ob(model_dir):
        df_valid = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/test.csv")
        trn2 = pd.read_csv('/kaggle/working/test_with_context_v2.csv')
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForMultipleChoice.from_pretrained(model_dir).cuda()
    
        predictions = []

        for index in tqdm(range(trn2.shape[0])):
            columns = df_valid.iloc[index].values
            question = columns[1]
            options = [columns[2], columns[3], columns[4], columns[5], columns[6]]
            context1 = trn2['context'][index]
            context2 = trn2['context_parsed'][index]
            context3 = trn2['context_wiki'][index]
            inputs1 = prepare_answering_input(
                tokenizer=tokenizer, question=question,
                options=options, context=context1,
                )
            inputs2 = prepare_answering_input(
                tokenizer=tokenizer, question=question,
                options=options, context=context2,
                )
            inputs3 = prepare_answering_input(
                tokenizer=tokenizer, question=question,
                options=options, context=context3,
                )

            with torch.no_grad():
                outputs1 = model(**inputs1)    
                losses1 = -outputs1.logits[0].detach().cpu().numpy()
                probability1 = torch.softmax(torch.tensor(-losses1), dim=-1)

            with torch.no_grad():
                outputs2 = model(**inputs2)
                losses2 = -outputs2.logits[0].detach().cpu().numpy()
                probability2 = torch.softmax(torch.tensor(-losses2), dim=-1)
            
            with torch.no_grad():
                outputs3 = model(**inputs3)
                losses3 = -outputs3.logits[0].detach().cpu().numpy()
                probability3 = torch.softmax(torch.tensor(-losses3), dim=-1)

            probability_ = probability1 * 0.4 + probability2 * 0.4 + probability3 * 0.2

            predictions.append(probability_.numpy())

        predictions = np.array(predictions)
    
        return predictions

Also, Inspired by [@simjeg](https://www.kaggle.com/simjeg)'s [notebook](https://www.kaggle.com/code/simjeg/platypus2-70b-with-wikipedia-rag) again, we made use of the variable `IS_TEST_SET` to save GPU run time for OpenBook model inference.

<br>

# Ensemble Models

To avoid overfitting the public LB, we simply took the average of the three OpenBook models. When ensembling the three non-OpenBook models, we assigned specific weights we derived from previous experience when playing with [@itsuki9180](https://www.kaggle.com/itsuki9180)'s [notebook](https://www.kaggle.com/code/itsuki9180/llm-sciex-optimise-ensemble-weights). After that, we assigned the OpenBook models to take 90% of the total prediction, while the non-OpenBook models could only take 10%. Finally we can ensemble the inference using the weights below:

    ws = [0.65 / 3, 0.65 / 3, 0.65 / 3, 0.12, 0.15, 0.08]
    ws = np.array(ws)

    openbook_w = 0.9 / 0.65
    other_w = 0.1 / 0.35

    ws[0] = ws[0] * openbook_w
    ws[1] = ws[1] * openbook_w
    ws[2] = ws[2] * openbook_w
    ws[3] = ws[3] * other_w
    ws[4] = ws[4] * other_w
    ws[5] = ws[5] * other_w

    predictions_overall = deberta_ob_preds_eric_0897 * ws[0] + deberta_ob_preds_eric_088 * ws[1] + deberta_ob_preds_eric_0916 * ws[2] + deberta_preds_billy_v1 * ws[3] + deberta_awp_preds_itk * ws[4] + deberta_lora_preds_lindsey * ws[5]

<br>

# Conclusion

In summary, what we mainly did in this competition was trying different kinds of ensembles, including ensembling different contexts and models. In those ensembles, we mainly contributed to training different models that were better than the ones posted publicly, and we borrowed those useful RAGs and brought them together with some fine-tuning. We were deeply surprised by the amazing open-source environment which helped everyone thrive in this competition. 

Finally, thanks again to all the competitors who shared those invaluable ideas that we could work on. We wouldn't get to this position without their effort. 

<br>

*P.S. Our score of 0.905 on the PB was achieved by a solution that did not use non-OpenBook models, it maintained the same weights for contexts ensemble but took the average of the three OpenBook models. However, the approach we posted here could achieve a score of 0.906, which is the best score on PB among all of our submissions. Hence we chose to post the best one publicly. Also, the score of 0.915 on the LB was achieved by a solution that is exactly the same as the one posted here, where we only changed to take the average of the three contexts.*
