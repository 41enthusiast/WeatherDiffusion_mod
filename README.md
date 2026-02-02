# Restoring Vision in Adverse Weather Conditions with Patch-Based Denoising Diffusion Models

This is the adapted code repository of the following [paper](https://arxiv.org/pdf/2207.14626.pdf) to train and perform inference with patch-based diffusion models for image restoration under adverse weather conditions.

Plans:
1. Dynamic programming to parallelize the grid overlapping process and create multiple workers.
2. Optimize the inference run time and training runtime with pytorch lightning wrapper
3. Mask aware logic - ideal vs predicted image scenario
4. Change from diffusion to flow model (WeatherFlow)

Current changes summarized:
1. gui for the whole process (needs polish)
2. one file all steps summarized for 1 image test code to borrow logic from.
3. requirements.txt for portability

"Restoring Vision in Adverse Weather Conditions with Patch-Based Denoising Diffusion Models"\
<em>Ozan Özdenizci, Robert Legenstein</em>\
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2023.\
https://doi.org/10.1109/TPAMI.2023.3238179

## Datasets

We perform experiments for combined image deraining and dehazing on [Outdoor-Rain](https://github.com/liruoteng/HeavyRainRemoval), and raindrop removal on
the [RainDrop](https://github.com/rui1996/DeRaindrop) datasets. To train multi-weather restoration, we used the AllWeather training set from [TransWeather](https://github.com/jeya-maria-jose/TransWeather), which is composed of subsets of training images from these three benchmarks.


## Saved Model Weights

We share a pre-trained diffusive **multi-weather** restoration model [WeatherDiff<sub>64</sub>](https://igi-web.tugraz.at/download/OzdenizciLegensteinTPAMI2023/WeatherDiff64.pth.tar) with the network configuration in `configs/allweather.yml`.

We also share our pre-trained diffusive multi-weather restoration model [WeatherDiff<sub>128</sub>](https://igi-web.tugraz.at/download/OzdenizciLegensteinTPAMI2023/WeatherDiff128.pth.tar) with the network configuration in `configs/allweather128.yml`.

To evaluate WeatherDiff<sub>64</sub> using the pre-trained model checkpoint with the current version of the repository:
```bash
python eval_diffusion.py --config "allweather.yml" --resume 'WeatherDiff64.pth.tar' --test_set 'raindrop' --sampling_timesteps 25 --grid_r 16
python eval_diffusion.py --config "allweather.yml" --resume 'WeatherDiff64.pth.tar' --test_set 'rainfog' --sampling_timesteps 25 --grid_r 16
python eval_diffusion.py --config "allweather.yml" --resume 'WeatherDiff64.pth.tar' --test_set 'snow' --sampling_timesteps 25 --grid_r 16
```





## Reference
Adapted the code from the paper:
```
@article{ozdenizci2023,
  title={Restoring vision in adverse weather conditions with patch-based denoising diffusion models},
  author={Ozan \"{O}zdenizci and Robert Legenstein},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  pages={1-12},
  year={2023},
  doi={10.1109/TPAMI.2023.3238179}
}
```

Parts of this code repository is based on the following works:

* https://github.com/ermongroup/ddim
* https://github.com/bahjat-kawar/ddrm
* https://github.com/JingyunLiang/SwinIR
