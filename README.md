# Face classification
![](similarity.png)
## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Convert facenet (embedding model) to tensorflow lite [here](https://colab.research.google.com/drive/1VovEl0I671JG7ufg2PtfjwKdM8YEK353?usp=sharing)

2. Download [dataset & model](https://drive.google.com/drive/folders/1y8CKhCWusiaZ3P86H5hZsW3gySTyLyaA?usp=sharing)
3. Train
```
python train.py
```
4. Test

4.1. multi-class classification
```
python inference.py
```
4.2. Binary classification
    
```
python image_similarity.py
```
example:

[![Watch the video](https://img.youtube.com/vi/HfN8CFpeV98/1.jpg)](https://youtu.be/HfN8CFpeV98)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)