# Semantic Search System - Test Suite

Bộ kiểm thử toàn diện cho hệ thống tìm kiếm ngữ nghĩa (Semantic Search System).

## Tổng quan

Bộ kiểm thử này bao gồm các test cases cho tất cả các thành phần chính của hệ thống tìm kiếm ngữ nghĩa:

- **Core Search Engine**: Kiểm thử chức năng tìm kiếm cơ bản
- **Similarity Algorithms**: Kiểm thử các thuật toán tính toán độ tương đồng
- **Ranking System**: Kiểm thử hệ thống xếp hạng kết quả
- **Filtering System**: Kiểm thử hệ thống lọc kết quả
- **Query Expansion**: Kiểm thử mở rộng và tiền xử lý truy vấn
- **Performance Optimization**: Kiểm thử caching và tối ưu hiệu suất
- **Integrated System**: Kiểm thử tích hợp end-to-end

## Cấu trúc thư mục

```
tests/
├── README.md                 # Tài liệu này
├── requirements-test.txt     # Dependencies cho testing
├── test_config.py           # Cấu hình và utilities cho testing
├── test_semantic_search.py  # Test suite chính
└── __init__.py
```

## Cài đặt Dependencies

### Cách 1: Sử dụng requirements file
```bash
pip install -r tests/requirements-test.txt
```

### Cách 2: Sử dụng test runner
```bash
python run_tests.py --install-deps
```

## Chạy Tests

### Sử dụng Test Runner (Khuyến nghị)

Test runner cung cấp nhiều tùy chọn để chạy tests:

```bash
# Chạy tất cả tests
python run_tests.py --all

# Chỉ chạy unit tests
python run_tests.py --unit

# Chỉ chạy integration tests
python run_tests.py --integration

# Chỉ chạy performance tests
python run_tests.py --performance

# Chạy test file cụ thể
python run_tests.py --file tests/test_semantic_search.py

# Chạy tests theo marker
python run_tests.py --marker similarity

# Chạy với verbose output
python run_tests.py --all --verbose

# Chạy mà không có coverage report
python run_tests.py --all --no-coverage

# Chạy code linting
python run_tests.py --lint

# Chạy type checking
python run_tests.py --type-check

# Tạo test report
python run_tests.py --report

# Dọn dẹp test artifacts
python run_tests.py --clean
```

### Sử dụng pytest trực tiếp

```bash
# Chạy tất cả tests
pytest tests/

# Chạy với coverage
pytest tests/ --cov=src --cov-report=html

# Chạy tests theo marker
pytest tests/ -m unit
pytest tests/ -m integration
pytest tests/ -m performance

# Chạy test cụ thể
pytest tests/test_semantic_search.py::TestSemanticSearchEngine::test_basic_search

# Chạy với verbose output
pytest tests/ -v

# Chạy parallel tests
pytest tests/ -n auto
```

## Test Markers

Hệ thống sử dụng các markers để phân loại tests:

- `unit`: Unit tests
- `integration`: Integration tests  
- `performance`: Performance tests
- `slow`: Tests chạy chậm
- `embedding`: Tests cần embedding models
- `vector_db`: Tests cần vector database
- `cache`: Tests cho caching functionality
- `similarity`: Tests cho similarity algorithms
- `ranking`: Tests cho ranking systems
- `filtering`: Tests cho filtering systems
- `query_expansion`: Tests cho query expansion
- `end_to_end`: End-to-end tests
- `mock`: Tests sử dụng mock services
- `real_services`: Tests sử dụng real services

## Cấu hình Test

### Test Configuration

File `test_config.py` chứa cấu hình cho test environment:

```python
@dataclass
class TestConfig:
    test_data_dir: str = "./test_data"
    temp_dir: str = "./temp_test"
    mock_embedding_dim: int = 384
    test_document_count: int = 100
    test_query_count: int = 50
    enable_performance_tests: bool = False
    enable_integration_tests: bool = True
```

### Pytest Configuration

File `pytest.ini` chứa cấu hình pytest:

- Test discovery patterns
- Coverage settings
- Markers definition
- Timeout settings
- Logging configuration

## Test Data

### Mock Data Generation

Hệ thống tự động tạo test data:

```python
from tests.test_config import setup_test_data

# Tạo test documents, queries và embeddings
documents, queries, embeddings = setup_test_data()
```

### Mock Services

Tạo mock services cho testing:

```python
from tests.test_config import create_test_services

# Tạo mock embedding service và vector database
embedding_service, vector_db_service = create_test_services(documents, embeddings)
```

## Test Categories

### 1. Unit Tests

Kiểm thử từng thành phần riêng lẻ:

- **SemanticSearchEngine**: Tìm kiếm cơ bản, lọc, xử lý lỗi
- **SimilarityAlgorithms**: Cosine, Euclidean, Hybrid, Adaptive
- **RankingSystem**: BM25, Hybrid scoring, Diversity reranking
- **FilteringSystem**: Content, Metadata, Temporal, Quality filters
- **QueryExpansion**: Preprocessing, Expansion methods
- **PerformanceOptimization**: Caching, Batch processing

### 2. Integration Tests

Kiểm thử tích hợp giữa các thành phần:

- **IntegratedSemanticSearch**: End-to-end search workflow
- **Service Integration**: Embedding + Vector DB + Search
- **Pipeline Integration**: Query → Search → Rank → Filter

### 3. Performance Tests

Kiểm thử hiệu suất:

- **Search Performance**: Thời gian response, throughput
- **Memory Usage**: Memory consumption, leak detection
- **Caching Performance**: Cache hit rates, performance improvement
- **Scalability**: Performance với large datasets

## Coverage Reports

### HTML Coverage Report

Sau khi chạy tests với coverage, mở file `htmlcov/index.html` để xem báo cáo chi tiết.

### Terminal Coverage Report

Coverage summary sẽ hiển thị trong terminal sau khi chạy tests.

### XML Coverage Report

File `coverage.xml` được tạo cho CI/CD integration.

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        python run_tests.py --install-deps
    
    - name: Run linting
      run: python run_tests.py --lint
    
    - name: Run type checking
      run: python run_tests.py --type-check
    
    - name: Run tests
      run: python run_tests.py --all --report
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Debugging Tests

### Chạy test với debugger

```bash
# Chạy với pdb
pytest tests/ --pdb

# Chạy với pdb on failures
pytest tests/ --pdb-trace
```

### Verbose logging

```bash
# Enable debug logging
pytest tests/ --log-cli-level=DEBUG
```

### Chạy test cụ thể

```bash
# Chạy method cụ thể
pytest tests/test_semantic_search.py::TestSemanticSearchEngine::test_basic_search -v

# Chạy class cụ thể
pytest tests/test_semantic_search.py::TestSimilarityAlgorithms -v
```

## Best Practices

### 1. Test Organization

- Mỗi test class tương ứng với một component
- Test methods có tên mô tả rõ ràng
- Sử dụng fixtures cho setup/teardown

### 2. Mock Usage

- Mock external services (embedding, vector DB)
- Sử dụng deterministic test data
- Isolate tests từ external dependencies

### 3. Performance Testing

- Sử dụng pytest-benchmark cho performance tests
- Set reasonable timeouts
- Monitor memory usage

### 4. Coverage

- Aim for >80% code coverage
- Focus on critical paths
- Test error conditions

## Troubleshooting

### Common Issues

1. **Import Errors**: Đảm bảo PYTHONPATH includes src directory
2. **Missing Dependencies**: Chạy `python run_tests.py --install-deps`
3. **Slow Tests**: Sử dụng `--maxfail=1` để stop on first failure
4. **Memory Issues**: Chạy tests với `--forked` option

### Debug Commands

```bash
# Check test discovery
pytest --collect-only tests/

# Run with maximum verbosity
pytest tests/ -vvv

# Show local variables on failure
pytest tests/ -l

# Show test durations
pytest tests/ --durations=10
```

## Contributing

Khi thêm tests mới:

1. Follow naming conventions
2. Add appropriate markers
3. Include docstrings
4. Update this README if needed
5. Ensure tests pass in CI

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)