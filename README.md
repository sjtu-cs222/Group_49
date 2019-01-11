# Group 49: Active Learning for Text Classification

#### Dependency Libraries

| Library | Version |
| ------- | ------- |
| Python  | 3.6.7   |
| Scipy   | 1.1.0   |
| Jieba   | 0.39    |
| Sklearn | 0.20.1  |
| Gensim  | 3.5.0   |



#### File dependencies

`Preprocessing.py ` : initialize and processes the input data , if you need to use the word2vec model

`construct.py`: different classifiers are defined here

`model.py`: main function entry implementing different kinds of active learning approaches

To run these algorithms

```
python model.py 
```



Parameters:

- initial_batch  --  set the initial labeled data size

- step		-- set the batch size
- model 	        -- set the classifier, e.g.: 'linear', 'logistic', 'bayes'
- methods        -- set the active sampling approaches, e.g.: 'active', 'margin', 'center' , 'random', 'entropy' ,'all' means running all these five algorithms
-  input_choice -- set the preprocessing methods, e.g.: 'tf-idf', 'word2vec' 
- plot                -- if true, then five graph of accuracy per batch are plotted, otherwise, accuracies are saved in a txt file

### Input data

`items.json`: taobao crawled review

`yanjing.anonymous.replace`: queried data




Finally, enjoy ur winter vacation www





