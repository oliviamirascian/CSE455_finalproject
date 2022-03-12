# Deblurring Anime Faces using Deep Convolutional Neural Networks 

* **You can find the dataset [here](https://www.kaggle.com/scribbless/another-anime-face-dataset) on Kaggle:**

* 1,000 images of the dataset is already in the `animefaces`  folder, to use the whole dataset extract it inside this folder

The directory structure for this project:

  ```
  ├───input
  │   ├───animefaces
  │   ├───box_filter_blurred
  │   ├───gaussian_blurred
  │   |───greyscaled
  │   └───motion_blurred
  ├───outputs
  │   ├───box_filter_deblurred
  │   ├───gaussian_deblurred
  │   └───motion_deblurred
  └───src
  ```

## How to Execute
 In `src` example using box blur
  1. `grayscale.py`
  2. `box_filter_blur.py`
  3. `train_box_filter.py`
 
## Example Result Images
The outputs are sorted in this order for each section (left to right, downwards):
* Original image
* Grayscaled image
* Blurred image
* Model 1 deblurred image
* Model 2 deblurred image
* Model 3 deblurred image

### <b> Box Filter Deblurring </b>
#### Learning Rate = 10<sup>-4</sup>:
<div align="center">
<figure>
 
  <img alt="box_filter_original" src="https://user-images.githubusercontent.com/39535587/157812715-19b7bca6-ab07-4fa0-ae4c-4795c2657b08.jpg">
  <img alt="box_filter_greyscaled" src="https://user-images.githubusercontent.com/39535587/157812466-1dca112a-eaa7-4e1f-bfd1-a11f9ee6b558.jpg">
  <img alt="box_filter_blurred" src="https://user-images.githubusercontent.com/39535587/157812327-5beb3374-8fab-4ed9-bf66-ed7de6dae4b0.jpg">

</figure>
 
</div>


<div align="center">
<figure>
  
  <img alt="box_filter_deblurred19_model1" src="https://user-images.githubusercontent.com/39535587/157813350-3f3a273e-8726-445c-85cb-e8cbc008b6b7.jpg">
  <img alt="box_filter_deblurred19_model2" src="https://user-images.githubusercontent.com/39535587/157813503-2266e469-7e3f-46af-bc10-e728cf583c83.jpg">
  <img alt="box_filter_deblurred19_model3" src="https://user-images.githubusercontent.com/39535587/157813515-1323c31c-1b2c-4312-b81b-a00c356ff219.jpg">

</figure>
</div>

**More on our website for this project [here](https://oliviamirascian.github.io/)**

## References 
1. C. Dong, C. C. Loy, K. He and X. Tang, "Image Super-Resolution Using Deep Convolutional Networks," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 38, no. 2, pp. 295-307, 1 Feb. 2016, doi: 10.1109/TPAMI.2015.2439281.
2. Jian Sun, Wenfei Cao, Zongben Xu, Jean Ponce. Learning a convolutional neural network for non-uniform motion blur removal. CVPR 2015 - IEEE Conference on Computer Vision and Pattern Recognition 2015, Jun 2015, Boston, United States. IEEE, 2015,.
3. Ledig, Christian & Theis, Lucas & Huszar, Ferenc & Caballero, Jose & Cunningham, Andrew & Acosta, Alejandro & Aitken, Andrew & Tejani, Alykhan & Totz, Johannes & Wang, Zehan & Shi, Wenzhe. (2017). Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. 105-114. 10.1109/CVPR.2017.19. 
4. Albluwi, Fatma & Krylov, Vladimir A. & Dahyot, Rozenn. (2018). Image Deblurring and Super-Resolution Using Deep Convolutional Neural Networks. 1-6. 10.1109/MLSP.2018.8516983. 
5. Kupyn, Orest & Budzan, Volodymyr & Mykhailych, Mykola & Mishkin, Dmytro & Matas, Jiri. (2017). DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks.
 
