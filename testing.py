import streamlit as st
import pickle

model = pickle.load(open('spam123.pkl', 'rb'))
cv = pickle.load(open('vec123.pkl','rb'))

def main():
  st.title("Email Spam Classification Application")
  st.write("This is a machine learning application to classify emails as spam or not spam")
  st.subheader("Classifiication")
  user_input = st.text_area("Enter your email here")
  if st.button("Predict"):
    if user_input:
      data = [user_input]
      print(data)
      vec = cv.transform(data).toarray()
      result = model.predict(vec)
      if result[0] == 0:
        st.success("Not Spam")
      else:
        st.error("Spam")
    else:
      st.warning("Please enter an email")

main()

