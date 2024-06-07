# LLM-based time series modeling of complex nonlinear dynamics

This repository is an extension of the paper:
[_Large Language Models Are Zero Shot Time Series Forecasters_](https://arxiv.org/abs/2310.07820)
by Nate Gruver, Marc Finzi, Shikai Qiu and Andrew Gordon Wilson (NeurIPS 2023).


<figure>
  <img src="./assets/tattoo.png" alt="Image">
  <figcaption> Schematic of our time series forecasting pipeline. This consists of three stages -- (a) Training of output projections with the generated samples at time points from dynamical systems (b) Training of the adapter of the LLM for fine-tuning (c) Using the trained model for probabilistic time series forecasting. </figcaption>
</figure>


## Contributors:
* Raunak Dey
* Chandramani Lu
* Ravi Chepuri
* You (contributions welcome)!


## 🛠 Installation
Run the following command to install all dependencies in a conda environment named `llmtime`. Change the cuda version for torch if you don't have cuda 11.8. 
```
source install.sh
```
After installation, activate the environment with
```
conda activate llmtime
```
If you prefer not using conda, you can also install the dependencies listed in `install.sh` manually. 

Add your openai api key to `~/.bashrc` with
```
echo "export OPENAI_API_KEY=<your key>" >> ~/.bashrc
```

Finally, if you have a diffferent OpenAI API base, change it in your `~/.bashrc` with
```
echo "export OPENAI_API_BASE=<your base url>" >> ~/.bashrc
```


## 💡 Tips 
Here are some tips for using LLMTime:
- Performance is not too sensitive to the data scaling hyperparameters `alpha, beta, basic`. A good default is `alpha=0.95, beta=0.3, basic=False`. For data exhibiting symmetry around 0 (e.g. a sine wave), we recommend setting `basic=True` to avoid shifting the data.
- The recently released `gpt-3.5-turbo-instruct` seems to require a lower temperature (e.g. 0.3) than other models, and tends to not outperform `text-davinci-003` from our limited experiments.
- Tuning hyperparameters based on validation likelihoods, as done by `get_autotuned_predictions_data`, will often yield better test likelihoods, but won't necessarily yield better samples. 


## Notebooks:

