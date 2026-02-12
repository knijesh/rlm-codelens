# API Key Security in RLM-Codelens

## 🔐 Current Implementation

### 1. Environment Variable Loading

API keys are loaded from environment variables via `.env` file:

```python
# src/rlm_codelens/core/config.py
class Config:
    def __init__(self):
        self.github_token: str = os.getenv("GITHUB_TOKEN", "")
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
```

### 2. Git Protection

`.env` file is excluded from git via `.gitignore`:
```
# .gitignore
.env
.env.local
.env.*.local
```

### 3. Validation

Configuration validates that keys are present:

```python
def validate(self) -> bool:
    errors = []
    if not self.github_token:
        errors.append("GITHUB_TOKEN is required")
    if not self.openai_api_key:
        errors.append("OPENAI_API_KEY is required")
    if errors:
        raise ValueError("Configuration errors: " + "; ".join(errors))
```

### 4. Secure Logging

Automatic redaction of sensitive data:

```python
from rlm_codelens.utils.secure_logging import get_logger

logger = get_logger(__name__)

# This will automatically redact the API key!
logger.info(f"Using token: {config.github_token}")
# Output: "Using token: [GITHUB_TOKEN_REDACTED]"
```

**Redaction patterns:**
- GitHub tokens: `ghp_...` → `[GITHUB_TOKEN_REDACTED]`
- OpenAI keys: `sk-...` → `[OPENAI_KEY_REDACTED]`
- Passwords: `password=secret` → `password=[REDACTED]`
- DB URLs: `postgresql://user:pass@...` → `postgresql://[USER]:[PASS]@...`

### 5. Safe String Representation

```python
from rlm_codelens.utils.secure_logging import safe_repr

# Instead of repr(config) which might expose keys:
print(safe_repr(config))
# Output: Config({
#   'github_token': '[REDACTED]',
#   'openai_api_key': '[REDACTED]',
#   'budget_limit': 50.0,
#   'rlm_model': 'gpt-3.5-turbo'
# })
```

## ✅ Security Checklist

- [x] API keys stored in environment variables
- [x] `.env` file in `.gitignore` (never committed)
- [x] No hardcoded keys in source code
- [x] Config validation ensures keys present
- [x] Automatic log redaction
- [x] Safe string representation
- [x] Error messages don't expose keys
- [x] Documentation for key management

## 🚀 Usage

### Development

```bash
# 1. Copy example file
cp .env.example .env

# 2. Edit with your keys
# Never commit this file!
nano .env

# 3. Validate configuration
python -c "from rlm_codelens.core.config import Config; Config().validate()"
```

### Production

```bash
# Set environment variables directly (no .env file)
export GITHUB_TOKEN=ghp_your_production_token
export OPENAI_API_KEY=sk_your_production_key
export BUDGET_LIMIT=100.0

# Run application
python main.py
```

### Testing

```python
# Use mock keys in tests
@pytest.fixture
def mock_config():
    config = Config()
    config.github_token = "ghp_test_token"
    config.openai_api_key = "sk_test_key"
    return config
```

## 🛡️ Best Practices Followed

1. **12-Factor App**: Configuration via environment variables
2. **Defense in Depth**: Multiple layers of protection
3. **Fail Secure**: Validation prevents running without keys
4. **Audit Trail**: Logging without exposing secrets
5. **Least Privilege**: Keys have minimal required permissions

## 📝 Key Rotation

**When to rotate:**
- Every 90 days (recommended)
- When team member leaves
- If key is accidentally exposed
- After security incident

**How to rotate:**
1. Generate new key
2. Update `.env` file
3. Test application
4. Revoke old key
5. Commit code changes (not `.env`!)

## 🚨 Incident Response

If you accidentally commit keys:

1. **Immediately revoke keys**
2. **Generate new keys**
3. **Update `.env` file**
4. **Clean git history** (if solo project):
   ```bash
   git filter-branch --force --index-filter \
   "git rm --cached --ignore-unmatch .env" \
   --prune-empty --tag-name-filter cat -- --all
   git push origin --force --all
   ```

⚠️ **Warning:** Force push can cause issues for collaborators!

## ✅ Verification

Test your security:

```bash
# 1. Check .env is gitignored
git check-ignore -v .env
# Should output: .gitignore:81:.env	.env

# 2. Verify no keys in git history
git log --all --full-history --source --name-only -- .env
# Should be empty

# 3. Check no keys in code
grep -r "sk-" src/ || echo "No OpenAI keys in code ✓"
grep -r "ghp_" src/ || echo "No GitHub tokens in code ✓"
```

## 📚 Additional Resources

- [docs/SECURITY.md](docs/SECURITY.md) - Complete security guide
- [GitHub Token Docs](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
- [OpenAI API Keys](https://platform.openai.com/docs/api-reference/authentication)
- [12-Factor App Config](https://12factor.net/config)
