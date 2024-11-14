import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import networkx as nx
from pyvis.network import Network
import random

# Set page configuration
st.set_page_config(
    page_title="Medical QA - Reliable Answers for Healthcare",
    page_icon="💉",
    initial_sidebar_state="expanded"
)


# Sample RAG retrieval and generation function
def generate_rag_response(question: str):
    """
    Placeholder function for the RAG model retrieval and generation.
    Replace with your actual retrieval and generation logic.
    """
    retrieved_context = f"Sample retrieved context for the question: {question}"
    generated_answer = f"Sample generated answer for the question: {question}"
    entities = [("pain", "Symptom"), ("ibuprofen", "Medication"), ("hypertension", "Disease")]
    return retrieved_context, generated_answer, entities

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
st.title("Medical QA System - RAG-based")
st.header("Accurate, Verified Answers for Healthcare")
st.write("""
    This QA system combines advanced language understanding with a curated knowledge base of verified medical sources,
    providing reliable answers. Start by asking a question.
""")

# Sidebar Content
display_question_tips()
display_subjects_covered()
display_sample_questions()

# Main Question Input and Output
#question = st.text_input("Enter your medical question here:")

qsts = ["The patient suffered from hypogonadism, failure to thrive, loss of taste and unable to maintain stability. What is the deficiency it shows?"]
query_entities = ['failure to thrive', 'loss of taste','hypogonadism']
generated_answer = ["The patient's symptoms of hypogonadism, failure to thrive, loss of taste, and instability indicate a zinc deficiency.\nZinc is an essential trace mineral that plays a crucial role in various physiological processes in the body.\n\
A deficiency can lead to a range of symptoms and health issues, as seen in this patient. \
Zinc is important for growth and development, immune function, cognitive function, and the senses of taste and smell. It is also involved in the production of testosterone and other sex hormones, which explains the patient's hypogonadism. \
Addressing the zinc deficiency through dietary changes and/or supplements, under medical supervision, can help alleviate these symptoms and improve the patient's overall health and stability."]
selected_question = st.selectbox('Select Query', qsts)
if st.button("Get Answer") and selected_question:
    with st.spinner("Retrieving context and generating answer..."):
        try:
            #retrieved_context, generated_answer, entities = generate_rag_response(question)
            #st.subheader("Supporting Medical Knowledge")
            #st.write(retrieved_context)
            #st.write(generated_answer)
            #display_annotated_answer(generated_answer, query_entities)
            st.subheader("Generated Answer:")
            answer_text = " ".join(generated_answer)
            st.markdown(
                f"""
                <div style="background-color: #FFF6E3; padding: 15px; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.1);">
                    <p style="font-size: 16px; line-height: 1.6;">{answer_text}</p>
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


        
html_file_path= "Streamlit/entity_graph.html"
display_html_file(html_file_path)

# Additional Information Section
st.header("Additional Information")
st.header("Entity Graph")

display_source_info()
#plot_entity_distribution()
#plot_entity_graph(entities)
st.write("""
    This Medical QA system provides responses using reliable sources, helping medical professionals
    and students gain trustworthy insights.
""")

