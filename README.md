# Hackathon: Exploring Foundation Models for Jet Physics

## The Context: Foundation Models in Science

In recent years, a growing area of exploration within the High Energy Physics (HEP)-ML community has focused on the potential of Foundation Models. Rather than training models for a single, narrow task, these research efforts investigate whether pretraining on massive, diverse datasets can yield universal representations of particle interactions that can benefit a wide variety of downstream physics tasks (like Anomaly Detection, new physics searches, classification) with potentially less data and training time.

## Our Focus: OmniLearned

This hackathon utilizes **OmniLearned**, a state-of-the-art foundation model architecture. We are working with fresh material; some of the material we are exploring is based on research released only a month ago:

1. [ArXiv:2510.24066](https://arxiv.org/abs/2510.24066)
2. [ArXiv:2603.23593](https://arxiv.org/abs/2603.23593)
3. [OmniLearned GitHub](https://github.com/ViniciusMikuni/OmniLearned)

Please refer to these resources throughout the hackathon.

### Acknowledgement

This hackathon is made possible by the open-source release of the OmniLearned framework. Special thanks to the development team (Wahid Bhimji, Chris Harris, Vinicius Mikuni, Benjamin Nachman) for their commitment to reproducible and accessible AI for the physical sciences.

---

## Environment Setup

```bash
conda create -n omni_env python=3.12
conda activate omni_env
pip install torch>=2.5.1
pip install omnilearned
conda install -c conda-forge jupyter notebook ipykernel
conda install -c conda-forge matplotlib
cd OmniLearned
```

---

## A. Top vs QCD Classification

### 1. Get the Data and Models

Download the dataset using the OmniLearned dataloader:

```bash
omnilearned dataloader --dataset top --folder ./omni_data
```

If the download is slow or fails, use `wget`:

```bash
wget -P ./omni_data/top/train https://portal.nersc.gov/cfs/dasrepo/omnilearned/top/train/train_ttbar.h5
wget -P ./omni_data/top/test  https://portal.nersc.gov/cfs/dasrepo/omnilearned/top/test/test_ttbar.h5
wget -P ./omni_data/top/val   https://portal.nersc.gov/cfs/dasrepo/omnilearned/top/val/val_ttbar.h5
```

**Note:** Due to limited compute resources, we will only work with the **small** pretrained model (3M parameters) and perform limited fine-tuning.

Download the pretrained checkpoint:
```bash
curl -L -O https://portal.nersc.gov/cfs/m4567/checkpoints/best_model_pretrain_s.pt
```

### 2. Baseline Evaluation

Evaluate the pretrained model's out-of-the-box performance on the top dataset:

```bash
omnilearned evaluate -i ../pretrained_checkpoints -o ./outputs \
    --save-tag pretrain_s --dataset top --path ./omni_data --size small \
    --interaction --local-interaction --num-classes 210 --use-event-loss --num-workers 1
```

Look at the paper [1] (specifically sections II A and C) to understand the `--interaction` and `--local-interaction` flags. This creates `outputs_pretrain_s_top_0.npz` in the `outputs/` folder.

As we don't have access to which of the 210 outputs is the top node, let's use the logic in `inspect_pretrained_output.ipynb` (Top node is one which has highest mean score for top events)
.

### 3. Full Fine-tuning

```bash
omnilearned train -o ./outputs --save-tag my_finetune --dataset top \
    --path ./omni_data --size small --epoch 2 --fine-tune --pretrain-tag pretrain_s \
    --interaction --local-interaction --num-workers 1
```

In this setup of finetuning, all the parameters are learnable so it takes quite a while (It took 5 hours for me on a GPU similar to what we have for hackathon, so run this at end of day, would be ready the next morning). 

The task-specific "head" learns 10x faster than the pre-trained body. You can control this with the `--lr-factor` flag. 

Once the process is done, you can run the evaluation:
```bash
omnilearned evaluate -i ./outputs -o ./outputs --save-tag my_finetune \
    --dataset top --path ./omni_data --size small --num-classes 2 \
    --interaction --local-interaction
```

and this will give you an output like what we saw in step 2. Compare step 3 output performance with that of step 2.


### 4. Faster Fine-tuning (Frozen Body)

For a faster alternative, freeze the model body and only train the classification head:

```bash
git clone https://github.com/abhi-mal/OmniLearned.git
# Switch to temp-testing branch for modified freezing logic
omnilearned evaluate -i ./outputs -o ./outputs --save-tag my_finetune \
    --dataset top --path ./omni_data --size small --num-classes 2 \
    --interaction --local-interaction
```

This would be much faster than step 3 (took 1.5 hours for me). Use a similar command like above to evaluate. Again compare the performance with step 2 and step 3. You can use  `compare_results.py` as reference. The plot I got is in `results/model_comparison_roc.png`.

### Analysis Questions

1. **Performance**: Is the performance difference between full and frozen fine-tuning what you expected? Why?
2. **Speed**: Why is step 4 faster than step 3?
3. **Architecture**: If we instead also let the input layers be learnable in addition to the task specific heads (keeping the bulk of model frozen), would it be just as fast as step 4? Why/ Why not?
4. **Physics Intuition**: We can define the top neuron as the neuron that has the highest score on average across top events in a dataset and similarly QCD neuron (what we did in inspect_pretrained_output.ipynb). Is the top neuron expected to be just as good a discriminator as the QCD neuron, for our problem of classifying top vs QCD process? 
Hint: Look at the right side plot in results/model_comparison_roc.png 
Why could this be happening?

<details>
<summary>Answer 4: Click to reveal</summary>

Because of the physics (assuming pretraining dataset is not skewed).
QCD == random spray of partons, can be 1, 2, 3, 4 or 5..
Top == spray with a particular structure mostly 3 or 1.

When a Top event comes in, it has a very "strong" signature, so the specific "Top neurons" fire with high confidence. QCD, on the other hand, is the "catch-all." Because it can look like almost anything (1-prong, 2-prong, wide, narrow), the probability mass is indeed smeared across many nodes.

This makes the Top neuron a better discriminator. It's much easier to say "This is definitely a Top".

</details>

---

## B. Anomaly Detection

OmniLearned supports multiple anomaly detection methods, including **CATHODE** and prong-based scoring. For this exercise, we will focus on CATHODE.

### What is CATHODE?

**CATHODE (Classifying Anomalies Through Outer Density Estimation)** is a "blind" anomaly detection method that discovers new physics without assuming a specific signal model. It works by comparing real data to a high-fidelity synthetic background.

**Workflow:**
- **Density Estimation**: A generative model (like what we worked with in the other hackathon leg) is trained on **Sideband (SB)** data to learn the distribution of known background.
- **Proxy Generation**: This model generates synthetic fake background events for the **Signal Region (SR)**.
- **Classification**: A classifier is trained to distinguish between **Real SR Data** (potentially containing signal) and the **Synthetic Fake Data** (pure background).
- **Discovery**: Events that the classifier identifies as "Real" with high confidence are the most anomalous.

**The Magic of CATHODE**: Because the Synthetic Fake Background contains 0 anomalies, the classifier learns that the only way to identify Real Data is by finding the BSM signal features.

**Note:** Generating synthetic background can take >20 hours. Look for pre-generated fake data files (`generated_*.h5`) in your dataset folders to skip the generation phase and jump straight to classification.

### Steps for Anomaly Detection

#### 1. Data Harvesting
```bash
omnilearned dataloader --dataset aspen_top_ad_sb --folder ./omni_data
omnilearned dataloader --dataset aspen_top_ad_sr --folder ./omni_data
```

#### 2. Density Estimation
```bash
omnilearned train -o ./outputs/AD --save-tag bg_gen \
    --dataset aspen_top_ad_sb --path ./omni_data \
    --interaction --local-interaction \
    --size small --mode generator --epoch 5 --num-workers 1
```

#### 3. Proxy Generation
```bash
omnilearned evaluate --mode generator --dataset aspen_top_ad_sr \
    --size small --save-tag bg_gen -i ./outputs/AD -o ./outputs/AD \
    --interaction --local-interaction \
    --path ./omni_data --num-workers 1
```

#### 4. Train the Anomaly Classifier
First, run the merging script to create the balanced dataset:
```bash
python merge_data.py
```

Then train the classifier:
```bash
omnilearned train -o ./outputs/AD --save-tag real_vs_fake \
    --dataset custom --path ./omni_data \
    --size small --num-classes 2 --epoch 10 \
    --interaction --local-interaction \
    --fine-tune --pretrain-tag pretrain_s --freeze
```

#### 5. Evaluation & Discovery
```bash
omnilearned evaluate -i ./outputs/AD -o ./outputs/AD --save-tag real_vs_fake_nersc \
    --dataset aspen_top_ad_sb --path ./omni_data --size small --num-classes 2 \
    --interaction --local-interaction
```

**Note:** In the paper, the authors show top quark rediscovery by full fine-tuning. As we are low on compute resource and because we saw that making only the classification head learnable gave a similar performance on Top vs QCD classification task, I tried doing the same for anomaly detection. But as you can see in `results/model_comparison_roc.png` this model wasn't as powerful in identifying the top quark.

### Additional Hackathon Goal
Can we reproduce the signal bump in the jet mass distribution using limited compute resources?

---

**Designed and maintained by:** Abhishikth Mallampalli
