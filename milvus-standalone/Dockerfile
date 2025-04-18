# Use official Python base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy your Streamlit app and requirements
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "usingMilvus.py"]
