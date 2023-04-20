import streamlit as st
import matplotlib.pyplot as plt
import cv2
import Harris_operator as harris
import Lambda
import SIFT
import FeatureMatching as fm
import time
st.set_page_config(layout="wide")

def body():
    col1, col2 = st.columns(2)
    with col1:
        st.header("Input Images")
    with st.sidebar:
        which_page = st.selectbox("", ["Harris Operator" , "Lambda Operator" , "SIFT Operator", "Feature Matching"])
        file = st.file_uploader("Upload file", type=["jpg","png"],key="file")
    if which_page=="Feature Matching":
        with st.sidebar:
            file2 = st.file_uploader("Upload second file", type=["jpg","png"],key="file2")
        if file:
            img_original = cv2.imread("images/{}".format(file.name))
            displayed_img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        
        if file2:
            img_original_2 = cv2.imread("images/{}".format(file2.name))
            displayed_img_original_2 = cv2.cvtColor(img_original_2, cv2.COLOR_BGR2RGB) 

        if file and file2:
            with st.sidebar:
                distance_metric = st.radio("Choose Distance Metric",('SSD', 'NCC'))
                image_quality = st.slider("Choose Image Quality (decreasing quality decreases computation time)", max_value=256, min_value=64, value=256 ,step=64)
            original = cv2.resize(img_original,(image_quality,image_quality))
            template = cv2.resize(img_original_2,(image_quality,image_quality))
            original_img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            template_img = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            start_time= time.time()
            kp1, des1 = SIFT.computeKeypointsAndDescriptors(original_img,save_img=0)
            kp2, des2 = SIFT.computeKeypointsAndDescriptors(template_img,save_img=0)
            matched_features = fm.feature_matching_temp(des1,des2,distance_metric)
            with st.sidebar:
                num_matches = st.slider("Choose Number Of Matches", min_value=10, max_value=len(matched_features), value=30,step=5)

            matched_features = sorted(matched_features, key=lambda x: x.distance, reverse=True)
            matched_image = cv2.drawMatches(original_img, kp1, template_img, kp2,matched_features[:num_matches], template_img, flags=2)
            end_time = time.time()
            plt.imshow(matched_image)
            plt.imsave("images/output.png",matched_image,cmap='gray')
            st.header(f"Computation time: {end_time - start_time}")
            with col1:
                col3,col4 = st.columns(2)
                with col3:
                    st.image(displayed_img_original, use_column_width=True)
                with col4:
                    st.image(displayed_img_original_2, use_column_width=True)
            with col2:
                st.header("Output Images")
                output_img = cv2.imread("images/output.png")
                displayed_output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                st.image(displayed_output_img, use_column_width=True)

    elif file:
        with col1:
            img_original = cv2.imread("images/{}".format(file.name))
            displayed_img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
            st.image(displayed_img_original, use_column_width=True)
        
        if which_page == "Harris Operator":
            with st.sidebar:
                start_time= time.time()
                harris.harris_corner_detector(img_original)
                end_time = time.time()
                    
          
        elif which_page == "Lambda Operator":
            with st.sidebar:
                start_time = time.time()
                Lambda.lambda_operator(img_original)
                end_time = time.time()
          
        elif which_page == "SIFT Operator":
            with st.sidebar:
                start_time = time.time()
                kp, ds= SIFT.computeKeypointsAndDescriptors(img_original)
                end_time = time.time()
     
        with col2:
            st.header("Output Images")
            output_img = cv2.imread("images/output.png")
            displayed_output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            st.image(displayed_output_img, use_column_width=True)
    
        st.header(f"Computation time: {end_time - start_time}")




if  __name__ == "__main__":
    body()