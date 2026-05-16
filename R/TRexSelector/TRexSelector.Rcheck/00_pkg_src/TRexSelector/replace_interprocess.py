import os

filepath = "../../src/utils/memmap/memory_mapped_matrix.cpp"
with open(filepath, "r") as f:
    content = f.read()

# Destructor replacement
old_dtor = """template<typename Scalar>
MemoryMappedMatrix<Scalar>::~MemoryMappedMatrix() {
    if (file_.is_open()) { file_.close(); }
}"""

new_dtor = """template<typename Scalar>
MemoryMappedMatrix<Scalar>::~MemoryMappedMatrix() {
    mapped_region_.reset();
    file_mapping_.reset();
}"""

content = content.replace(old_dtor, new_dtor)

# Initialize replacement
old_init = """    // 2. Open the Mapping
    boost::iostreams::mapped_file_params params;
    params.path = filename_;
    params.length = required_bytes;

    if (mode_ == AccessMode::ReadOnly) {
        params.flags = boost::iostreams::mapped_file::readonly;
    } else {
        params.flags = boost::iostreams::mapped_file::readwrite;
    }

    try {
        file_.open(params);
    } catch (const std::exception& e) {
        throw std::runtime_error("Boost map failed: " + std::string(e.what()));
    }

    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open memory map: " + filename_);
    }"""

new_init = """    // 2. Open the Mapping
    try {
        boost::interprocess::mode_t b_mode = (mode_ == AccessMode::ReadOnly) 
            ? boost::interprocess::read_only 
            : boost::interprocess::read_write;
            
        file_mapping_ = std::make_unique<boost::interprocess::file_mapping>(filename_.c_str(), b_mode);
        mapped_region_ = std::make_unique<boost::interprocess::mapped_region>(*file_mapping_, b_mode, 0, required_bytes);
    } catch (const std::exception& e) {
        throw std::runtime_error("Boost map failed: " + std::string(e.what()));
    }"""

content = content.replace(old_init, new_init)

# Access hints replacement
old_hints = """template<typename Scalar>
void MemoryMappedMatrix<Scalar>::applyAccessHints() {
    if (!file_.is_open()) return;

#if !defined(_WIN32) && !defined(_WIN64)
    // MADV_SEQUENTIAL tells the OS to aggressively read-ahead
    // MADV_RANDOM would disable read-ahead
    madvise(file_.data(), file_.size(), MADV_SEQUENTIAL);
#endif
}"""

new_hints = """template<typename Scalar>
void MemoryMappedMatrix<Scalar>::applyAccessHints() {
    if (!mapped_region_ || !mapped_region_->get_address()) return;

#if !defined(_WIN32) && !defined(_WIN64)
    // MADV_SEQUENTIAL tells the OS to aggressively read-ahead
    // MADV_RANDOM would disable read-ahead
    madvise(mapped_region_->get_address(), mapped_region_->get_size(), MADV_SEQUENTIAL);
#endif
}"""

content = content.replace(old_hints, new_hints)

# Getters replacement
content = content.replace("file_.data()", "mapped_region_->get_address()")
content = content.replace("file_.const_data()", "mapped_region_->get_address()")

with open(filepath, "w") as f:
    f.write(content)
