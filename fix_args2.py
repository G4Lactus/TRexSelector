import re
with open("tests/test_utils/test_utils_datagen.cpp", "r") as f:
    text = f.read()

# Replace the specific tests we added to use the correct constructor args
text = re.sub(r'EXPECT_THROW\(\(trex::utils::datageneration::datagen::SyntheticData\s*\(\s*0, 10, 0, 1\.0, 5, pred_policy, trex::utils::datageneration::dummygen::Distribution::Normal\(\), noise_policy, 42\)\), std::invalid_argument\);',
    r'EXPECT_THROW((trex::utils::datageneration::datagen::SyntheticData(\n        0, 10, 0, {0, 1}, {1.0, 1.0}, 1.0, 42, -1, -1, pred_policy, trex::utils::datageneration::dummygen::Distribution::Normal(), noise_policy)), std::invalid_argument);', text, flags=re.MULTILINE)

text = re.sub(r'EXPECT_THROW\(\(trex::utils::datageneration::datagen::SyntheticData\s*\(\s*10, 0, 0, 1\.0, 5, pred_policy, trex::utils::datageneration::dummygen::Distribution::Normal\(\), noise_policy, 42\)\), std::invalid_argument\);',
    r'EXPECT_THROW((trex::utils::datageneration::datagen::SyntheticData(\n        10, 0, 0, {0, 1}, {1.0, 1.0}, 1.0, 42, -1, -1, pred_policy, trex::utils::datageneration::dummygen::Distribution::Normal(), noise_policy)), std::invalid_argument);', text, flags=re.MULTILINE)

with open("tests/test_utils/test_utils_datagen.cpp", "w") as f:
    f.write(text)
