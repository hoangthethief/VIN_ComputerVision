import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import os
from urllib.parse import urlparse
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import spacy
from model.caption_vu import CombinationModel
from model.resnet50 import ResNet50

text_embedder = spacy.load('en_core_web_lg')
model = CombinationModel().to('cpu')
# model = ResNet50(15).to('cpu')
model.load_state_dict(torch.load('checkpoint\\best_checkpoint.model'))
model.eval()

def get_caption(image_path):
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    return preds

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("City Image Classifier")
st.text("Provide URL of City Image for image classification")

# @st.cache(allow_output_mutation=True)
def load_image_data(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    caption = get_caption(image_path)[0]
    caption = text_embedder(caption).vector

    transform = A.Compose([
                            A.SmallestMaxSize(max_size=360),
                            A.CenterCrop(height=256, width=256),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensorV2()])

    img = transform(image=img)['image']

    return img, caption


# with st.spinner('Loading Model Into Memory...'):
#     model = load_model()

classes = ['Newyork', 'Singapore', 'Sydney', 'Venezia', 'Amsterdam', 'Roma',
                        'Moscow', 'Hanoi', 'Rabat', 'Kyoto', 'Dubai', 'Rio', 'Maldives',
                        'Paris', 'London']

path = st.text_input('Enter Image URL to Classify..', 'https://a.travel-assets.com/findyours-php/viewfinder/images/res70/69000/69643-Kyoto.jpg')
if path is not None:
    content = requests.get(path).content

    a = urlparse(path)

    filename = "web_data/" + os.path.basename(a.path)

    r = requests.get(path, allow_redirects=True)

    i = Image.open(BytesIO(content))
    st.image(i, caption="Classifying City Image", use_column_width = True)

    open(filename, 'wb').write(r.content)
    

    image, caption = load_image_data(filename)
    image = image[None, :].to('cpu')
    caption = caption[None, :]
    caption = torch.from_numpy(caption).to('cpu')

    st.write("Predicted class: ")
    with st.spinner('classifying.....'):
        predict = model(image, caption)
        # predict(image)
        label = np.argmax(predict.detach().numpy(), axis = 1)
        new_title = f'<p style="font-family:sans-serif; color:Green; font-size: 42px;">{classes[label[0]]}</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        # st.write(classes[label[0]])
    st.write('')
    
