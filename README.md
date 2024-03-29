# A Real-time Vision Framework for Pedestrian Behavior Recognition and Intention Prediction at Intersections Using 3D Pose Estimation


We propose a real-time (> 30 FPS) vision framework for two central tasks in intelligent transportation systems:

- Pedestrian Behavior Recognition
- Crossing or Not-Crossing Intention Prediction

Below represents the overall architecture of the proposed vision framework.

<p float="center">
  <img src="data/figures/overall_architecture.png" width="95%" />
</p>

Receiving a sequence of image frames, the proposed framework 1) extracts both 3D and 2D pose of pedestrian of interest in addition to V2P and environment contexts, 2) analyzes the behavior of the pedestrian and 3) predicts the intention of crossing or not-crossing.

<p float="center">
  <img src="data/figures/pose_eg_2d.png" width="49%" />
  <img src="data/figures/pose_eg_3d.png" width="49%" />
</p>

Our project includes the following software packages

- [Multi-object tracker](multi_object_tracker/mot.py)
- [Human pose analyzer (projection of 3D pose to 2D imaging planes)](demo_2d_3d_joints.py)
- [Feature (V2P &amp; environmental contexts) extractor](demo_feature_extraction.py)
- [Trainer and tester for intention prediction](intention_prediction/main.py) [(+ raw data)](intention_prediction/data_raw)

> [**A Real-time Vision Framework for Pedestrian Behavior Recognition and Intention Prediction at Intersections Using 3D Pose Estimation**](https://arxiv.org/abs/2009.10868),
> [Ue-Hwan Kim](https://github.com/Uehwan), [Dongho Ka](https://stslabblog.wordpress.com/people/), [Hwasoo Yea](https://stslabblog.wordpress.com/people/), [Jong-Hwan Kim](http://rit.kaist.ac.kr/home/jhkim/Biography_en),
> *IEEE Transactions on Intelligent Transportation Systems, Under Review, 2020*

## Getting Started

We implemented and tested our framework on Ubuntu 18.04 with python >= 3.6. It supports both GPU and CPU inference.

Clone the repo:

```bash
git clone https://github.com/Uehwan/VisionForPedestrian.git
```

Install the requirements using `virtualenv` or `conda`:

```bash
# pip
source scripts/install_pip.sh

# conda
source scripts/install_conda.sh
```

## Running the Demos

### 3D and 2D Pose Estimation: Behavior Analysis

Simply run the following:

```bash
python demo_2d_3d_joints.py
```

### Semantic Segmentation

Simply run the following:

```bash
python demo_semantic_segmentation.py
```

### Multi-Object Tracking and Feature Extraction

#### 1. Videos to Images

First, you need to extract images from video files by running

```bash
python demo_video_processing.py --root_dir PATH_TO_ROOT --output_dir PATH_TO_OUTPUT
```

The root_dir should look like this (this is to support processing of multiple videos at once):

```bash
|---- ROOT_DIR
|     |---- folder_1
|           |---- video_1.mp4
|           |---- video_2.mp4
|           |---- video_3.mp4
|     |---- folder_2
|           |---- video_1.mp4
```

After processing, the output_dir becomes the following structure:

```bash
|---- OUTPUT_DIR
|     |---- folder_1
|           |---- video_1
|                 |---- 000001.png
|                 |---- 000002.png
|                 |---- 000003.png
|           |---- video_2
|                 |---- 000001.png
|                 |---- 000002.png
|                 |---- 000003.png
|           |---- video_3
|                 |---- 000001.png
|                 |---- 000002.png
|                 |---- 000003.png
|     |---- folder_2
|           |---- video_1
|                 |---- 000001.png
|                 |---- 000002.png
|                 |---- 000003.png
```

#### 2. Detectors and Trackers

Next, you need to run detectors and object trackers.
Before running detectors and trackers, prepare vibe data by executing

```bash
source scripts/prepare_vibe_data.sh
```

Then,

```bash
python demo_detection.py --root_dir PATH_TO_ROOT --output_dir PATH_TO_OUTPUT
```

Here, the root_path is same as the output_dir of the previous step (demo_video_processing)

#### 3. Labeling Crosswalks

Before you extract features for intention prediction, you need to label crosswalk positions by running

```bash
python demo_label_crosswalk.py --root_dir PATH_TO_ROOT
```

To label the entrance of each crosswalk, click two ends of each crosswalk sequentially. Then, the script will automatically save the labeling result. For example, click cw1-endpoint1 => cw1-endpoint2 => cw2-endpoint1 => cw2-endpoint2 => cw3-endpoint1 => ...

(You can run "demo_crosswalk.py" for the automatic crosswalk detection with tensorflow > 2.1, but the performance is not satisfactory)

If you're working on a single scene, you can simply insert the crosswalk position in the [demo_feature_extraction.py](demo_feature_extraction.py) file

#### 4. Identifying Same Pedestrians and Labeling Signal Phase

Since the object detector and tracker are not perfect, you need to label the same object ids.

Provide the same pedestrian labels as the following format (xlsx or csv in one column) for each video file and put them in the 'data_annotation' folder (you could name them as 'date_id_pedestrian.xlsx').

```
p_id_match
1,3,5
10,15,22,55
32
59
```

For signal phases, follow the below format for each video and put them in the 'data_annotation' folder (name them as 'date_id_signal.xlsx').
```
signal_phase    frame
Green           1558
Flashing Green  1888
Red	            2308
Green           2966
Flashing Green  3296
Red             3716
Green           6989
```

In the above example, the signal is green for frames in the range of (0, 1558].

#### 5. Extract Pedestrian and Vehicle Features

Then, run the below

```bash
python demo_feature_extraction.py --root_dir DIR_TO_DATA_PICKLE
```

After extracting features, divide them into 'train', 'val', and 'test' sets. For this, make three directories ('train', 'val', and 'test' inside 'data_csv') and put each csv in the corresponding to each split directory.

### Training and Testing the Performance of Intention Prediction

```bash
# For other configurations, refer to experiment_graph.sh & experiment_table.sh
cd intention_prediction
python main.py \
    --exp_name "GRU_ST_CSO_l2_h32_F05_CL05" \
    --model_type "gru" \
    --num_layers 2 \
    --hidden_size 32 \
    --future_stamps 0.5 \
    --num_output 1 \
    --context_length 0.5
```

You can run the following to retrieve the evaluation results reported in our manuscript.

```bash
python plot_results.py
```

## Evaluation Results

<p float="center">
  <img src="data/figures/intention_FFNN_ST.png" width="32%" />
  <img src="data/figures/intention_gru_ST.png" width="32%" />
  <img src="data/figures/intention_transformer_ST.png" width="32%" />
</p>

<p float="center">
  <img src="data/figures/intention_FFNN_MT.png" width="32%" />
  <img src="data/figures/intention_gru_MT.png" width="32%" />
  <img src="data/figures/intention_transformer_MT.png" width="32%" />
</p>

## Citation

If you find this project helpful, please consider citing this project in your publications. The following is the BibTeX of our work.

```bibtex
@inproceedings{kim2020a,
  title={A Real-time Vision Framework for Pedestrian Behavior Recognition and Intention Prediction at Intersections Using 3D Pose Estimation},
  author={Kim Ue-Hwan, Ka Dongho, Yeo Hwasoo, Kim Jong-Hwan},
  journal = {arXiv preprint arXiv:2009.10868},
  year = {2020}
}
```

## License

This code is available for **non-commercial scientific research purposes**. Third-party datasets and software are subject to their respective licenses.

## Acknowledgments

We base our project on the following repositories:

- 3D Pose Estimation: [VIBE](https://github.com/mkocabas/VIBE)
- Multiple People Tracking: [MPT](https://github.com/mkocabas/multi-person-tracker)
- Object Detecctor: [Yolov3](https://github.com/mkocabas/yolov3-pytorch)
- Semantic Segmetation: [HRNetV2](https://github.com/CSAILVision/semantic-segmentation-pytorch)

This work was supported by Institute for Information & communications Technology Promotion (IITP) grant funded by the Korea government (MSIT) (No.2020-0-00440, Development of Artificial Intelligence Technology that Continuously Improves Itself as the Situation Changes in the Real World)
