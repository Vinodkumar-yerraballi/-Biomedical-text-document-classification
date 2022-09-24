# Import the standard librarys for the model deploment
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import time
import base64
import streamlit.components.v1 as components

# Set background title
st.set_page_config(
    page_title="BIO Medical CALSSIFIER APP",
    layout="wide",
    initial_sidebar_state="expanded"
)

# set background image
# set the background color


@st.cache(persist=True, show_spinner=False)
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: 100 cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background(
    '/home/vinod/anaconda3/envs/datascience/Biomedical/istockphoto-1311222341-170667a.jpg')
# Title of the website
st.title("THE EMAIL CLASSIFICICTION")
# load the dataset using the pandas
data = pd.read_csv('alldata_1_for_kaggle.csv', encoding='latin1')
# Rename the column names
data.rename({'0': 'Label', 'a': 'Text'}, axis=1, inplace=True)
# Covert the data categorical data to numrecial using map function
data['Label'] = data['Label'].map(
    {'Thyroid_Cancer': 0, 'Colon_Cancer': 1, 'Lung_Cancer': 2})
# divide the data into x and y variable
x = data['Text']
y = data['Label']
# Let's covert the text data into arrys using the TfidfVectorizer
vector = TfidfVectorizer()
vector.fit(x)
x = vector.transform(x)

# And define the model for predict the test data and fit the x and y variables
model = LogisticRegression()
model.fit(x, y)

# define a function for receive the text data from the users


def main():
    # recevie from the input message from the users
    text = st.text_area("Enter your  Biomedical text hear",
                        'Enter your text........')
    # Define the button
    st.button("Predict")
    # condition statement if the text length minimum 1 return to text otherwise do prediction
    if len(text) < 1:
        st.write("")
    else:
        # the text message must be dimension that's way we create in list
        text = [text]
        # after we transfom the text data into arrays using the TdifVectorize
        text_int = vector.transform(text)
        # We predict the user input is using the xgbclassifier
        prediction = model.predict(text_int)
        # And finaly we print the prediction using the success function
        with st.spinner("Wait for it Checking the  Biomedical text 'ThyroidCancer','ColonCancer','Lung_Cancer'..."):
            time.sleep(5)
        if prediction == 0:
            st.success("Biomedical text document classification Thyroid_Cancer")
            st.balloons()
        if prediction == 1:
            st.success("Biomedical text document classification Colon_Cancer")
            st.balloons()
        else:
            st.success("Biomedical text document classification Lung_Cancer")
            st.balloons()
        # html componets
    components.html("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Biomedical text document classification </title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
</head>
<body>
    <div>
    <br>
    <br>
    <br>
    <br>
    <ul>
        <li>
    <a href="https://github.com/Vinodkumar-yerraballi">
    <i class="fa fa-github" style="font-size:48px;color:red "></i></a>       </li>
    <br>
    <br>
      <li>
    <a href="https://www.linkedin.com/in/vinod-kumar-yerraballi-44520214b/">
    <i class="fa fa-linkedin" style="font-size: 48px;color:blue"></i>
    </a>
    </li>
    </ul>
    </div>
    
</body>
</html>
""", height=600)


# and define the function
if __name__ == '__main__':
    main()
