## Papers

***

### Evaluating Prose Style Transfer with the Bible

* _Source_ :https://arxiv.org/ftp/arxiv/papers/1711/1711.04731.pdf

#### Parellel vs. non-parallel data
* Highly parallel data with the goal of transferring style via supervised learning. Big part of this paper is presenting the parallel data corpus over learning methods
* If you want to generalized style instead of specific feature, parallel data needed
* Note that there is work in unparallel, unsupervised learning - author’s opinion tends to perform worse than when parallel data is available - still not 100% needed to surpervised

#### Discussion on sources
* Simple wikipedia another popular parallel corpus. Word embeddings also used to inform alignment. Newsela has reading level parallel data. Grammerly YAFC constructed form Yahoo answers then turned into more formal language by mechanical turk
* Shakespeare plays and their modern translations another dataset with large number of aligned sentences.

#### Seq2Seq
* The Seq2Seq model was first created and used in conjunction with statistical methods to perform machine translation [29]. The model consists of a recurrent neural network acting as an encoder, which produces an embedding of the full sequence of inputs. This sentence embedding is then used by another recurrent neural network which acts as a decoder and produces a sequence corresponding to the original input sequence. Seq2Seq model was adapted to use multiple LSTM layers on both the encoding and decoding sides.  Seq2Seq model requires a fixed vocabulary which contains the tokens which will be encountered by the model.
* Minibatches of 64 verse-pairs, truncated to 100 tokens, Adam optimizer

#### Moses model
* Look into the Moses model for statistical machine translation - author trained it five times in total for different styles

#### Evaluation
* BLEU / PINC scores for evaluation. Seq2Seq always had higher PINC score than Moses but outperforms on BLEU.

#### Conclusion
* It is likely that some previously published modifications to Seq2Seq would result in immediate performance improvements. Candidates from the machine translation literature include: coverage modelling [52] to help track which parts of the source sentence have already been paraphrased and the use of a pointer network [53] to allow copying of words directly from the source sentence. Pointer networks have already been used for style transfer [11], and seem likely to be useful for our multi-style corpus as well.

***

### Disentangled Representation Learning for Non-Parallel Text Style Transfer
* _Source_ : https://arxiv.org/pdf/1808.04339.pdf

#### Latent space vs. style space
* Style and content are indeed disentangled in the latent space. This disentangled latent representation learning method is applied to style transfer on non-parallel corpora. We achieve substantially better results in terms of transfer accuracy, content preservation and language fluency, in comparison to previous state-of-the-art approaches
* We artificially divide the latent representation into two parts: the style space and content space. In this work, we consider the sentiment of a sentence as the style. We design auxiliary losses, enforcing the separation of style and content latent spaces.

#### Non-parallel text style transfer
* We call this *non-parallel text style transfer*. To accomplish this, we train an autoencoder with disentangled latent spaces. For style-transfer inference, we simply use the autoencoder to encode the content vector of a sentence, but ignore its encoded style vector. We then infer from the training data, an empirical embedding of the style that we would like to transfer. The encoded content vector and the empirically-inferred style vector are concatenated and fed to the decoder.

#### Autoencoder model
* Our model is built upon an autoencoder with a sequence-to-sequence neural network and we design multi-task and adversarial losses for both style and content spaces. The encoder recurrent neural network (RNN) with gated recurrent units (GRU); the autoencoder is trained with a sequence-aggregated cross-entropy loss.
* The autoencoding losses in Equations (1,2) serve as our primary training objective. We also design several auxiliary losses to disentangle the latent space. They build a softmax applied to the style vector as a style classfier, trained alongside autoencoder - multi-task learning. Also apply an adversarial loss to discourage the content space from containing style information - The idea is to first introduce a classifier, called an *adversary*, that deliberately discriminates the true style label using the content vector c.

#### Cost function
* Continue with a content specific loss, while the overall cost function comprises several terms: the reconstruction objective, the multi-task objectives for style and content, and the adversarial objectives for style and content with hyperparameters to balance their weightings.

#### Data
* Ran model on Yelp and Amazon reviews, which come with sentiment labels - used to train the latent space and to evaluate sentiment transfer. Used Adam for autoencoder and RMSProp for discriminators. 

#### Conclusion
* Metrics: Style transfer - trained a separate CNN to predict sentiment, then used that classifier as the truth for the style transfer. Content - cosine similarity, word overlap, KL smoothed language model, manual eval. 

***

### Style Transfer in Text: Exploration and Evaluation

* _Source_: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17015/15745

#### Idea: use adversarial networks
* In response to the challenge of lacking parallel data, we explore learning style transfer from non-parallel data. The key idea behind the proposed models is to learn separate content representations and style representations using adversarial networks.

#### Transfer strength and content preservation
* We propose two novel evaluation metrics that measure two aspects of style transfer: transfer strength and content preservation. We benchmark our models and the evaluation metrics on two style transfer tasks: paper-news title transfer, and positive-negative review transfer.

#### Seq2Seq success on parallel data (machine translation)
* Sequence to sequence (seq2seq) neural network models (Sutskever, Vinyals, and Le 2014) have demonstrated great success in many generation tasks, such as machine translation, dialog system and image caption, with the requirement of a large amount of parallel data. However, it is hard to get parallel data for tasks such as style transfer.

#### Models
* We explore two models for text style transfer, to approach the aforementioned problems of 1) lacking parallel training data and 2) hard to separate the style from the content.
    1. The first model implements a __multi-decoder seq2seq__ where the encoder is used to capture the content c of the input X, and the multi-decoder contains n(n ≥ 2) decoders to generate outputs in different styles.
        * The multi-decoder model for style transfer is similar to an auto-encoder with several decoders 
        * Exception that the encoder now tries to learn some content representations that do not reflect styles 
        * The style specific decoders (one for each style) then take the content representations and generate texts in different styles.
    
    2. The second model uses the same encoding strategy, but introduces __style embeddings that are jointly trained__ with the model. The style embeddings are used to augment the encoded representations, so that only one decoder needs to be learned to generate outputs in different styles.
        * Our second model uses style embeddings to control the generated styles. 
        * The encoder and the adversarial network parts are the same as the multi-decoder model, to generate content representations.  
        * A single decoder is trained in this model, which takes the concatenation of the content representation c and the style embedding e of a sentence as the input to generate texts in different styles.
    
#### Auto-encoder model
* In auto-encoder seq2seq model, both the encoder and decoder are recurrent neural networks (RNNs). We employ the gated recurrent unit (GRU) variant which uses gates to control the information flow. 

#### Evaluation and overhead
* Evaluation: transfer strength - We use a LSTM-sigmoid classifier which performs well in big data, evaluating sentiment using a binaruy classifier. Content preservation - cosine distance, for word embeddings used pre-trained Glove 
* Performed some preprocessing to cut down on difficult data, used paper news title dataset and pos/neg review dataset.

***

### Collecting Highly Parallel Data for Paraphrase Evaluation

* _Source_ : https://www.aclweb.org/anthology/P11-1020.pdf

#### Framework
1. First, we describe a framework for easily and inexpensively crowdsourcing arbitrarily large training and test sets of independent, redundant linguistic descriptions of the same semantic content. 
2. Second, we define a new evaluation metric, PINC (Paraphrase In N-gram Changes), that relies on simple BLEU-like n-gram comparisons to measure the degree of novelty of automatically generated paraphrases.
* Used Mech. Turk to collect summaries of videos etc. 

#### Conclusion
* Paper more useful for eval metrics - all three scores are combined using a support vector machine (SVM) trained on human ratings of paraphrase pairs.  
* PEM, which uses a second language as pivot to establish semantic equivalence. To measure semantic equivalence, we simply use BLEU with multiple references. 
* PINC that measures how many n-grams differ between the two sentences. 
* In essence, it is the inverse of BLEU since we want to minimize the number of n-gram overlaps between the two sentences.

***

### Shakespearizing Modern Language

* _Source_ : https://www.aclweb.org/anthology/W17-4902.pdf. 
* _Code_ : https://github.com/harsh19/Shakespearizing-Modern-English

#### Method
* To tackle limited amount of parallel data, we _pre-train embeddings of words_ by leveraging external dictionaries mapping Shakespearean words to modern English words as well as additional text.
* Our dataset is a collection of line-by-line modern paraphrases for 16 of Shakespeare’s 36 plays (*Antony & Cleopatra*, *As You Like It*, *Comedy of Errors*, *Hamlet*, *Henry V* etc) from the educational site *Sparknotes*

#### Model
* We use a _bidirectional LSTM_ to encode the input modern English sentence. 
* Our decoder side model is a mixture model of RNN module amd pointer network module. 
* The decoder RNN predicts probability distribution of next word over the vocabulary, while pointer model predicts probability distribution over words in input.

#### Comparison
* Each token in vocabulary is represented by a M dimensional embedding vector. 
    * Note that we do not directly use off-the-shelf pretrained embeddings such as *Glove* (Pennington et al., 2014) and *Word2Vec* (Mikolov et al., 2013) since we need to learn embeddings for novel word forms
* A pair of corresponding *Original* and *Modern* sentences have significant vocabulary overlap.
    * Moreover, there are lot of proper nouns and rare words which might not be predicted by a sequence to sequence model. 
    * To rectify this, pointer networks have been used to enable copying of tokens from input directly
    
#### Training
* Cross entropy loss is used to train the model. 
* Sentinel Loss (SL): Following from work by (Merity et al., 2016), we consider additional sentinel loss. 
    * This loss function can be considered as a form of *supervised attention*.
* We lowercase sentences and then use NLTK’s PUNKT tokenizer to tokenize all sentences.

#### Evaluation
* Our primary evaluation metric is *BLEU* (Papineni et al., 2002) . We compute *BLEU* using the freely available and very widely used perl script7 from the MOSES decoder. 
* We also report *PINC* 

#### Implementation
* We use a minibatch-size of 32 and the *ADAM* optimizer (Kingma and Ba, 2014) with learning rate 0.001, momentum parameters 0.9 and 0.999, and ε = 10−8. All our implementations are written in Python using Tensorflow 1.1.0 framework.

***

### Problems in Current Text Simplification Research

* _Source_ : https://cocoxu.github.io/publications/tacl2015-text-simplification-opinion.pdf

#### Method
* We introduce a new simplification dataset that is a significant improvement over _Simple Wikipedia_, and present a novel quantitative-comparative approach to study the quality of simplification data resources.


* The __Parallel Wikipedia Simplification (PWKP)__ corpus prepared by Zhu et al. (2010), has become the benchmark dataset for training and evaluating automatic text simplification systems. However, we will show that this dataset is deficient and should be considered obsolete. 
    1) It is prone to automatic sentence alignment errors; 
    2) It contains a large proportion of inadequate simplifications; 
    3) It generalizes poorly to other text genres.
    
#### Newsela
* Newsela: Each article has been re-written 4 times for children at different grade levels by editors at Newsela2, a company that produces reading materials for pre-college classroom use. 
* We use Simp-4 to denote the most simplified level and Simp-1 to denote the least simplified level.
* It is motivated by the Common Core Standards (Porter et al., 2011) in the United States. All the Newsela articles are grounded in the _Lexile3 readability score_, which is widely used to measure text complexity and assess students’ reading ability

***

### Neural Style Transfer for Non-Parallel Text

* _Source_ : https://github.com/vinitra/neural-text-style-transfer/blob/master/reports/Neural_Style_Transfer_for_Non_Parallel_Text.pdf

#### Relation to CSAIL on non-parallel text
* Draws heavily on MIT CSAIL’s Style Transfer from Non-Parallel Text by Cross Alignment: https://arxiv.org/pdf/1705.09655.pdf. Another paper to check out
* They build on this work by trying to copy exact authorship style and propose new evaluation metrics

#### CSAIL
* CSAIL took discriminator NN that tries to predict  the sentence style the data came from  - train autoencoder generator to minimize content loss while maximizing error of the discriminator to strip sentence of identifiable style
* Then add in third decoder RNN to output the content in the target style.

#### Data
* Collected 1M tweets from celebrities and high profile accounts, cut down to 10 accounts with strong authorship style

#### Architecture
* Architecture - RNN GRU to find content latent variable, one layer FF NN to be discriminator, duplicate encoder to act as decoder to learn content representation. 
* Then optimize decoder to output target style.

#### Evaluation
* Used author classifier to determine success of style transfer. For content presertation send the sentence round trip twice to see how close original and reconverted sentence are  - BLEU score on those sentences.

***

### Neural Text Style Transfer via Denoising and Reranking

* _Source_ : https://www.aclweb.org/anthology/W19-2309.pdf
* Note Andrew Ng is a coauthor, so I suppose we can ask him questions need be.

#### Method
* Frames style transfer as denois-ing: we synthesize a noisy corpus and treat the source style as a noisy version of the target style.
    * Start with non-parallel data in different styles 
    * Then synthesize parallel data using backtranslation 
    * Produce noise in a way that should mimic the styles of the opposite text
    
#### Use in parallelized text
* With the parallel corpus, train a denoising NN between noisy and clean data - prepend each sentence with a token denoting its style. Train the model using both sets of noised / clean data - target and source styles.
* Use a transformer encoder-decoder with byte-pair encoding. 

#### Evaluation
* Evaluation: We define transfer strength as the ratio of target-domain to source-domain shift in sentence probability. 
    * Content - use cosine similarity. 
    * Fluency - We use the average log probability of the sentence post-transfer with respect to a language model trained on CommonCrawl as our measure of fluency.
    
#### Conclusion
* Our method relies on an initial seed corpus of(clean, noisy)sentence pairs to bootstrap training.  
* However, such a corpus is _not ideal_ for the style transfer tasks we consider, as there is mismatch in many cases between the style transfer domains (e.g.news, music lyrics, forum posts) and the seed corpus (language learner posts).



