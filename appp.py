from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st
import os

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check mortgage eligibility and provide reasons if ineligible
def check_mortgage_eligibility(details: dict) -> dict:
    income = details.get("income")
    credit_score = details.get("credit_score")
    loan_amount = details.get("loan_amount")
    property_value = details.get("property_value")
    
    reasons = []
    eligible = True

    # Check income eligibility
    if income < 30000:
        reasons.append(f"Your income of {income} is less than the required 30,000 INR per month.")
        eligible = False

    # Check credit score eligibility
    if credit_score < 650:
        reasons.append(f"Your credit score of {credit_score} is below the required 650.")
        eligible = False

    # Check loan amount eligibility
    if loan_amount > 0.8 * property_value:
        reasons.append(f"The loan amount of {loan_amount} exceeds 80% of the property value.")
        eligible = False

    if eligible:
        return {
            "eligible": True,
            "message": "Congratulations! You are eligible for a mortgage loan.",
            "suggestions": []
        }
    else:
        suggestions = []
        if credit_score < 650:
            suggestions.append("Consider improving your credit score to at least 650.")
        if loan_amount > 0.8 * property_value:
            suggestions.append(f"Consider applying for a smaller loan amount (max {0.8 * property_value}).")
        if income < 30000:
            suggestions.append("Consider increasing your monthly income to meet the 30,000 INR minimum.")
        
        return {
            "eligible": False,
            "message": "Sorry, based on the provided details, you are not eligible for a mortgage loan.",
            "reasons": reasons,
            "suggestions": suggestions
        }

# Function to generate responses using ChatGroq
def get_response(user_query: str, user_details: dict, next_step: str):
    eligibility_info = check_mortgage_eligibility(user_details)
    
    # If the user asks "why" or for an explanation, provide the reasons
    if "why" in user_query.lower() or "explain" in user_query.lower():
        if not eligibility_info["eligible"]:
            return "Here’s why you're ineligible: " + "; ".join(eligibility_info["reasons"])
        else:
            return "You're already eligible for a mortgage loan."

    # Provide suggestions to improve eligibility
    if "suggest" in user_query.lower():
        return "Here are some suggestions: " + "; ".join(eligibility_info["suggestions"])

    if next_step == "eligibility_check":
        return eligibility_info["message"]
    
    # Continue the normal conversation
    template = """
    You are an assistant helping users determine if they are eligible for a mortgage loan based on the details they provided.
    
    User provided the following details so far:
    - Income: {income}
    - Credit Score: {credit_score}
    - Loan Amount: {loan_amount}
    - Property Value: {property_value}
    
    Your task is to continue the conversation, ask the user for more details step by step, and determine eligibility.
    
    Next Step: {next_step}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0, api_key=GROQ_API_KEY)

    chain = (
        RunnablePassthrough.assign(schema=user_details)
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "income": user_details.get("income", "Not provided yet"),
        "credit_score": user_details.get("credit_score", "Not provided yet"),
        "loan_amount": user_details.get("loan_amount", "Not provided yet"),
        "property_value": user_details.get("property_value", "Not provided yet"),
        "next_step": next_step,
    })

# Streamlit app setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm here to help you check your mortgage loan eligibility. Please enter your details below."),
    ]
    st.session_state.user_details = {}
    st.session_state.next_step = "get_details"

st.set_page_config(page_title="Mortgage Loan Checker", page_icon=":house:")

st.title("MortgageMate")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# User input form for mortgage details
with st.form("mortgage_form", clear_on_submit=True):
    income = st.number_input("Income (INR)", min_value=0.0)
    credit_score = st.number_input("Credit Score", min_value=0, max_value=850)
    loan_amount = st.number_input("Desired Loan Amount (INR)", min_value=0.0)
    property_value = st.number_input("Property Value (INR)", min_value=0.0)
    
    submitted = st.form_submit_button("Check Eligibility")

if submitted:
    st.session_state.user_details = {
        "income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "property_value": property_value,
    }
    response = get_response("Check eligibility", st.session_state.user_details, "eligibility_check")
    st.session_state.chat_history.append(AIMessage(content=response))
    with st.chat_message("AI"):
        st.markdown(response)

# Explanation and suggestions buttons
if st.session_state.chat_history[-1].content and "not eligible" in st.session_state.chat_history[-1].content.lower():
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Why am I ineligible?"):
            explanation = get_response("Why am I ineligible?", st.session_state.user_details, "eligibility_check")
            st.session_state.chat_history.append(AIMessage(content=explanation))
            with st.chat_message("AI"):
                st.markdown(explanation)
    
    with col2:
        if st.button("Suggestions for Improvement"):
            suggestions = get_response("suggest", st.session_state.user_details, "eligibility_check")
            st.session_state.chat_history.append(AIMessage(content=suggestions))
            with st.chat_message("AI"):
                st.markdown(suggestions)
