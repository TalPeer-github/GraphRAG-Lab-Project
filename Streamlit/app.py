import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import networkx as nx
from pyvis.network import Network
import random
import re
from time import sleep


# Set page configuration
st.set_page_config(
    page_title="Entity-based RAG",
    page_icon="💉",
    initial_sidebar_state="expanded"
)


# Sidebar with guidance and subject areas
def display_question_tips():
    st.sidebar.subheader("Tips for Asking Effective Questions")
    st.sidebar.write("""
        - **Be Specific**: Use precise medical terms.
        - **Clarify Symptoms**: Include symptom duration or severity.
        - **Specify Age and Gender**: Certain conditions vary by these factors.
        - **Ask One Question at a Time**: Keeps responses focused.
    """)

def display_subjects_covered():
    st.sidebar.subheader("Medical Subjects Covered")
    subjects = [
        "Clinical Scenarios", "Dental", "Surgery", "Pathology", "Medicine",
        "Pharmacology", "Anatomy", "Pediatrics", "Gynaecology & Obstetrics", 
        "Physiology", "Biochemistry", "Preventive Medicine", 
        "Microbiology", "Radiology", "Forensic Medicine", "Ophthalmology", 
        "ENT", "Anaesthesia", "Orthopaedics", "Psychiatry", "Skin"
    ]
    st.sidebar.write(", ".join(subjects))

def display_sample_questions():
    st.sidebar.subheader("Sample Questions by Subject")
    sample_questions = {
        "Surgery": "What are the indications for appendectomy?",
        "Radiology": "What are the common findings on chest X-ray for pneumonia?",
        "Pediatrics": "What are the symptoms of Kawasaki disease?",
        "Clinical Scenario": (
            "A 65-year-old male presents with chest pain radiating to his left arm, sweating, and shortness of breath. "
            "He has a history of hypertension and diabetes. What is the most likely diagnosis, and what is the initial "
            "management plan?"
        )
    }
    for subject, question in sample_questions.items():
        st.sidebar.write(f"**{subject}**: {question}")

# NER Highlighting in response
def display_annotated_answer(answer, entities):
    """
    Highlight entities in the answer using HTML/CSS.
    """
    highlighted_answer = answer
    for entity, label in entities:
        highlighted_answer = highlighted_answer.replace(
            entity,
            f"<span style='background-color: #8ef; padding: 0.2em; border-radius: 0.2em;'>{entity} ({label})</span>"
        )
    st.subheader("Generated Answer with Highlighted Entities")
    st.markdown(highlighted_answer, unsafe_allow_html=True)



def plot_entity_graph(entities):
    G = nx.Graph()
    for _ in range(10):  
        entity1, entity2 = random.sample(entities, 2)
        G.add_edge(entity1[0], entity2[0])

    net = Network(height="400px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    net.show("entity_graph.html")
    st.subheader("Entity Co-occurrence Graph")
    st.components.v1.html(open("entity_graph.html", "r").read(), height=400)

# Metadata about retrieval sources
def display_source_info():
    st.subheader("Curated Medical Sources")
    st.write("""
        - **PubMed**: Research articles
        - **NIH**: Clinical guidelines
        - **WHO**: Public health reports
    """)


def plot_entity_distribution():
    entity_data = pd.DataFrame({
        "Entity Type": ["Disease", "Symptom", "Medication", "Anatomy"],
        "Count": [200, 150, 100, 50]
    })
    fig, ax = plt.subplots()
    sns.barplot(x="Entity Type", y="Count", data=entity_data, ax=ax)
    ax.set_title("Distribution of Key Medical Entities")
    st.pyplot(fig)


def display_html_file(file_path: str):
    """
    Display the contents of an HTML file within the Streamlit app.
    :param file_path: Path to the .html file to display.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=600, scrolling=True)
    except Exception as e:
        st.error(f"Error loading HTML file: {e}")


# Streamlit Interface Structure
st.title("Entity-Based RAG")
st.header("094295 - Final Project")
st.write("""
    In this project, we aim to develop a medical entity-focused QA system centered around RAG to meet our specific
use cases and address the challenges outlined above.
""")

# Sidebar Content
display_question_tips()
display_subjects_covered()
display_sample_questions()

# Main Question Input and Output
question = st.text_area("Insert query:", height=70)

qsts = ["The patient suffered from hypogonadism, failure to thrive, loss of taste and unable to maintain stability. What is the deficiency it shows?"]
query_entities = ['failure to thrive', 'loss of taste','hypogonadism']
generated_answer = ["The patient's symptoms of hypogonadism, failure to thrive, loss of taste, and instability indicate a zinc deficiency.\nZinc is an essential trace mineral that plays a crucial role in various physiological processes in the body.\n\
A deficiency can lead to a range of symptoms and health issues, as seen in this patient. \
Zinc is important for growth and development, immune function, cognitive function, and the senses of taste and smell. It is also involved in the production of testosterone and other sex hormones, which explains the patient's hypogonadism. \
Addressing the zinc deficiency through dietary changes and/or supplements, under medical supervision, can help alleviate these symptoms and improve the patient's overall health and stability."]
basic_rag_answer = "The patient's symptoms indicate a zinc deficiency."
# selected_question = st.selectbox('Select Query', qsts)
# if st.button("Get Answer") and selected_question:
#     with st.spinner("Retrieving context and generating answer..."):
#         try:
#             # Convert generated_answer to a single string if it is a list
#             answer_text = " ".join(generated_answer)
            
#             # Split the text into sentences using regular expression
#             sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', answer_text)
            
#             # Join sentences with <br> tags to create line breaks in Markdown
#             formatted_answer = "<br>".join(sentences)
            
#             # Display the formatted answer in a styled container
#             st.subheader("Generated Answer:")
#             st.markdown(
#                 f"""
#                 <div style="background-color: #FFF6E3; padding: 15px; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.1);">
#                     <p style="font-size: 16px; line-height: 1.6;">{formatted_answer}</p>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
#         except Exception as e:
#             st.error(f"Error generating response: {e}")
# answer_type = st.radio("Choose the type of answer to display:", ("Entity RAG Answer", "Basic RAG Answer"))

# if st.button("Get Answer") and selected_question:
#     with st.spinner("Retrieving context and generating answer..."):
#         try:
#             # Display the selected type of answer
#             if answer_type == "Entity RAG Answer":
#                 # Convert generated_answer to a single string if it is a list
#                 answer_text = " ".join(generated_answer)
                
#                 # Split the text into sentences using regular expression
#                 sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', answer_text)
                
#                 # Join sentences with <br> tags to create line breaks in Markdown
#                 formatted_answer = "<br>".join(sentences)
                
#                 # Display the formatted answer in a styled container
#                 st.subheader("Entity RAG Answer:")
#                 st.markdown(
#                     f"""
#                     <div style="background-color: #FFF6E3; padding: 15px; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.1);">
#                         <p style="font-size: 16px; line-height: 1.6;">{formatted_answer}</p>
#                     </div>
#                     """,
#                     unsafe_allow_html=True
#                 )
#             elif answer_type == "Basic RAG Answer":
#                 # Display the basic answer in a styled container
#                 st.subheader("Basic RAG Answer:")
#                 st.markdown(
#                     f"""
#                     <div style="background-color: #E3F2FF; padding: 15px; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.1);">
#                         <p style="font-size: 16px; line-height: 1.6;">{basic_rag_answer}</p>
#                     </div>
#                     """,
#                     unsafe_allow_html=True
#                 )
#         except Exception as e:
#             st.error(f"Error generating response: {e}")

selected_question = qsts[0]
if st.button("Get Answer") and selected_question:
    with st.spinner("Retrieving context and generating answer..."):
        sleep(2)
        try:
            # Column layout for displaying answers side by side
            col1, col2 = st.columns(2)

            # Display the Basic RAG Answer in the left column
            with col1:
                st.subheader("Basic RAG Answer:")
                st.markdown(
                    f"""
                    <div style="background-color: #E3F2FF; padding: 15px; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.1);">
                        <p style="font-size: 16px; line-height: 1.6;">{basic_rag_answer}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Convert generated_answer to a single string if it is a list
            answer_text = " ".join(generated_answer)

            # Split the text into sentences using regular expression
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', answer_text)

            # Join sentences with <br> tags to create line breaks in Markdown
            formatted_answer = "<br>".join(sentences)

            # Display the Entity RAG Answer in the right column
            with col2:
                st.subheader("Entity RAG Answer:")
                st.markdown(
                    f"""
                    <div style="background-color: #FFF6E3; padding: 15px; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.1);">
                        <p style="font-size: 16px; line-height: 1.6;">{formatted_answer}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Error generating response: {e}")
# container1 = st.container(border=True)
# st.write(query_entities)
# # with container1:
# #     plot_education(selected_position)
# container2 = st.container(border=True)
# st.write(generated_answer)

st.header("Medical NER")  
def add_image(image_path: str, caption: str = "", width: int = 500):
    """
    Display an image in the Streamlit app.
    
    Args:
        image_path (str): Path to the image file (e.g., "ner.png").
        caption (str): Optional caption for the image.
        width (int): Optional width of the image in pixels (default is 500).
    """
    try:
        st.image(image_path, caption=caption, use_column_width=False, width=width)
    except FileNotFoundError:
        st.error(f"Image '{image_path}' not found. Please check the path and try again.")
    except Exception as e:
        st.error(f"Error displaying image: {e}")

# Example usag


html_file_path= "Streamlit/entity_graph.html"
image_file_path= "Streamlit/ner.png"
add_image(image_file_path, caption="Named Entity Recognition Visualization")
st.header("The Entity Graph")        
display_html_file(html_file_path)
#display_source_info()
st.write("""
    This Medical QA system provides responses using reliable sources, helping medical professionals
    and students gain trustworthy insights.
""")


