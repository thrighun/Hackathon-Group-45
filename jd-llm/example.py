from inference import JDModel

jd = JDModel()

# Step 1 — Generate
result = jd("Python developer, 2 years exp, Django, REST APIs, Bangalore")
print(result)

# Step 2 — Edit
result = jd("add Docker and Kubernetes to requirements")
print(result)

# Step 3 — Reset and start new
jd.reset()