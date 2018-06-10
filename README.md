# Writeup
Udacity 자율주행 코스의 [프로젝트 과제](https://github.com/udacity/CarND-Vehicle-Detection)를 위해 vehicle detection을 수행하였다. 여기서 소개하는 결과물은 많은 부분 해당 코스의 소스코드와 writeup template 문서를 참고하였음을 미리 밝혀둔다.

---

[//]: # (Image References)
[image1]: ./output_images/binned_color_feature.png
[image2]: ./output_images/histogram_color_featrue.png
[image3]: ./output_images/hist_n_bin.png
[image4]: ./output_images/color_feature_training_result.png
[image5]: ./output_images/hog_feature.png
[image6]: ./output_images/hog_feature_training_result.png
[image7]: ./output_images/all_feature_training_result.png
[image8]: ./output_images/search.png
[image9]: ./output_images/heatmap.png
[video1]: ./project_video.mp4

## Vehicle Detection Project 
최근 트랜드로 보면 딥러닝 기반 object detection 기법이나 sementic segmentation이 전통적인 방식의 feature extraction 기반 방법론 보다 더 좋은 성능을 보이고 있다. 하지만 딥러닝 접근법과 전통적인 접근법 모두 장단점이 있으므로 전통적인 방법론도 알아두면 좋을 것이다. 여기에서는 feature extraction과 SVM 분류기, sliding window의 조합으로 오브젝트를 추출하는 전통적인 방법론을 사용하여 vehicle detection을 수행해본다. 

아래와 같은 단계를 수행할 것이다.
- Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
- Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
- Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
- Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
- Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
- Estimate a bounding box for vehicles detected.

## 1. Feature Extraction

### Color Features
색상 피쳐를 위해서는 아래와 같이 두 가지 시도를 하였다. 
* binned color features: 32x32, 3 channel 
![alt text][image1]
* color histogram features 
![alt text][image2]
* 두 가지 색상 피쳐를 단순 concatenate하여 최종 색상 feature vector를 생성한다. 이때 아래 그림과 같이 적절하게 normalize 해주어야 분류기 학습할 때 성능 저하를 막을 수 있다. 
![alt text][image3]

최종적으로 색상 피쳐만을 이용해서 분류기를 학습했을 때 대략 95% 정도의 정확도를 얻을 수 있었다. (LinearSVC, RGB)
![alt text][image4]

### HOG Features
색상 피쳐는 오브젝트의 형태 정보를 잡아낼 수 없기 때문에 HOG 알고리즘도 추가로 활용한다. 
![alt text][image5]

HOG 피쳐를 이용해서 분류기를 학습해서 아래와 같이 대략 98% 이상의 정확도를 얻을 수 있었다. (LinearSVC, SVH 3 channel, orient=9, pix_per_cell=8, cell_per_block=2)
![alt text][image6]

### 모두 사용
위에서 사용한 모든 피쳐를 조합하면 최종적으로 99% 이상의 정확도를 가지는 자동차 이미지 분류기를 얻을 수 있다. (YCrCb)
![alt text][image7]

## 2. Sliding Window Search
자동차에 탑재된 카메라에서 얻은 영상으로부터 자동차에 해당하는 영역을 추출하기 위해서 슬라이딩 윈도우 방식을 사용한다. 슬라이딩 윈도의의 각 이미지 패치를 위에서 만든 분류기의 입력으로 사용하여 자동차인지 여부를 판단한다. 총 3가지 스케일의 윈도우를 사용하였고 각 스케일 별로 대상 추출 영역을 다르게 하여 효율화를 꾀하였다. 
![alt text][image8]

이렇게 얻은 윈도우가 많이 중첩되는 영역을 자동차가 위치한 영역이라고 판단하는 방식으로 최종 자동차의 bounding box를 얻는다. 
![alt text][image9]

## 3. Video Implementation
위에서 작성한 이미지 처리 파이프라인을 비디오 영상에 적용해보았다. 결과 영상은 [여기](https://youtu.be/nut9yFeYKUI)에서 찾을 수 있다. False positive 문제를 해결하기 위해 현재 이미지 프레임에서 자동차로 분류된 영역이 이전 이미지 프레임에서도 자동차로 인식된 경우에만 자동차 영역으로 인식하도록 하였다. 
<div align="left">
  <a href="https://www.youtube.com/watch?v=nut9yFeYKUI"><img src="https://img.youtube.com/vi/nut9yFeYKUI/0.jpg" alt="IMAGE ALT TEXT"></a>
</div>
