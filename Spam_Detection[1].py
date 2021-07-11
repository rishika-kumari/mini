import pickle 
import streamlit as st


nb_classifier = pickle.load(open("Spam_Detection.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))


def main():
	st.title("Spam Classification App")
	st.subheader("Built with Streamlit And Python")
	msg = st.text_input("Enter The Text : ")
	if st.button("Predict"):
		data = [msg]
		vect = vectorizer.transform(data).toarray()
		prediction = nb_classifier.predict(vect)
		result = prediction[0]
		if result == 1 :
			st.error("This is a Spam")
		else :
			st.success("This is a Ham")
main()