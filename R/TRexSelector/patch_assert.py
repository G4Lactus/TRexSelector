path = "src/tsolvers/tsolver_base.cpp"
with open(path, "r") as f:
    text = f.read()

text = text.replace("    assert(budget > 0);", "    if (budget <= 0) throw std::runtime_error(\"Budget must be > 0\");")

with open(path, "w") as f:
    f.write(text)
