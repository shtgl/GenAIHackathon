# Bajaj Finserv GenAI Hackathon

<div align="justify">
The following repo consists of files which extract data from pdf and provided context based response.
</div>

## Setup
<h3>
<div>
<img src="img/env.png"; width=20> <b>Environment</b>
</div>
</h3>

*Note - The above codes are written using Python virtual environment version **3.12.1** and supports the latest Python versions. However, lower Python versions may offer more stability compared to the latest ones. Below setup will focus Windows system. And commands to setup may vary for macOS or Linux.*



<div> 1. Create a python virtual environment using -

```bash
# Path to installation of particular version of python  may vary
# I have installed more than one version of python in pyver directory
# Refer the resources section, for youtube video to install multiple versions of python
C:\Users\<username>\pyver\py3121\python -m venv brainenv
```
</div>


<div>2. Activate the virtual environment using -

```bash
brainenv/Scripts/activate
```
</div>

<div> 3. Install python packages using - 

```bash
pip install -r requirements.txt
```
</div>

<div> 4. Jupyter Notebook Configuration -

```bash
ipython kernel install --user --name=myenv
python -m ipykernel install --user --name=myenv
```
</div>

<div> 5. For bash based cell execution inside Jupyter (Optional) -

```bash
pip install bash_kernel
python -m bash_kernel.install
```
</div>

<div> 6. Congratulations! Now the environment is ready for execution -

```bash
jupyter notebook
```
</div>


