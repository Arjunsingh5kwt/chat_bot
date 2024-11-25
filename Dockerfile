# Step 1 Install the dependencies
FROM python:3.10-slim

# Step 2 Working Directory
WORKDIR /app

# Step 3 copy the file
# COPY . /app

# Install dependencies
COPY requirements.txt .

#step 4 
RUN pip install --no-cache-dir -r requirements.txt

# Install Uvicorn if not present
RUN pip install uvicorn

# Copy the current directory contents into the container at /app
COPY . .

# step 5
EXPOSE 8000

# step 6 Set the command
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

