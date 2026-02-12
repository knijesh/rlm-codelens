# API Key Security Best Practices

## 🔐 Current Implementation

Your project already follows good practices:
- ✅ API keys stored in `.env` file (not in code)
- ✅ Loaded via environment variables
- ✅ `.env` is in `.gitignore` (won't be committed)
- ✅ Validation ensures keys are present
- ✅ No hardcoded keys in source code

## 🛡️ Enhanced Security Recommendations

### 1. **Environment Variable Management**

```bash
# Create .env file (already done)
cp .env.example .env

# Edit with your actual keys (never commit this file!)
nano .env
```

### 2. **Key Rotation Strategy**

**GitHub Token:**
- Create token at: https://github.com/settings/tokens
- Use minimal scopes: `repo` (for public repos) or `repo, read:org` (for private)
- Rotate every 90 days
- Never share tokens

**OpenAI API Key:**
- Create at: https://platform.openai.com/api-keys
- Use separate keys for dev/prod
- Set spending limits on OpenAI dashboard
- Monitor usage regularly

### 3. **Production Deployment**

**For Production Servers:**
```bash
# Use system environment variables (not .env file)
export GITHUB_TOKEN=ghp_your_token_here
export OPENAI_API_KEY=sk-your_key_here

# Or use a secrets manager like AWS Secrets Manager, Azure Key Vault, or HashiCorp Vault
```

**Docker Deployment:**
```dockerfile
# Don't put keys in Dockerfile!
# Pass at runtime:
# docker run -e GITHUB_TOKEN=$GITHUB_TOKEN -e OPENAI_API_KEY=$OPENAI_API_KEY your-image
```

### 4. **Logging Safety**

Ensure API keys are NEVER logged:

```python
# ❌ BAD: Logging the entire config
logger.info(f"Config: {config}")  # Might leak keys!

# ✅ GOOD: Log only non-sensitive data
logger.info(f"Budget: ${config.budget_limit}")
logger.info(f"Model: {config.rlm_model}")
```

### 5. **Error Message Safety**

```python
# ❌ BAD: Including keys in error messages
raise ValueError(f"Invalid API key: {api_key}")

# ✅ GOOD: Generic error messages
raise ValueError("Invalid API key provided")
```

## 🔍 Security Checklist

- [ ] `.env` file is in `.gitignore`
- [ ] No API keys committed to git history
- [ ] Keys use minimal required permissions
- [ ] Production uses environment variables (not .env)
- [ ] Keys are rotated regularly
- [ ] Spending limits are set on OpenAI dashboard
- [ ] Error messages don't expose keys
- [ ] Logs don't contain sensitive data
- [ ] Team members don't share keys
- [ ] Keys are revoked when team members leave

## 🚨 If You Accidentally Commit Keys

1. **Immediately revoke the keys:**
   - GitHub: https://github.com/settings/tokens → Delete token
   - OpenAI: https://platform.openai.com/api-keys → Delete key

2. **Create new keys** with same permissions

3. **Update your .env file** with new keys

4. **Clean git history** (if recently committed):
   ```bash
   git filter-branch --force --index-filter \
   "git rm --cached --ignore-unmatch .env" \
   --prune-empty --tag-name-filter cat -- --all
   ```

5. **Force push** (only if working alone!):
   ```bash
   git push origin --force --all
   ```

⚠️ **WARNING:** Force pushing rewrite history can cause issues for collaborators!

## 📝 Testing Without Real Keys

For testing, use mock keys:

```python
# In test fixtures (conftest.py)
@pytest.fixture
def mock_config():
    return Config()
    # Override with test values
    config.github_token = "ghp_test_token"
    config.openai_api_key = "sk_test_key"
    return config
```

## 🔒 Additional Security Measures

1. **Use IP Whitelisting** (if possible)
2. **Enable 2FA** on GitHub and OpenAI accounts
3. **Monitor API usage** for anomalies
4. **Set up alerts** for high spending
5. **Use separate keys** for different environments
6. **Implement rate limiting** in your application

## 🎯 Summary

Your current implementation is **GOOD** ✅

**What's already secure:**
- Environment variable loading
- .env file excluded from git
- Validation of required keys
- No hardcoded secrets

**What you should add:**
- Regular key rotation
- Production deployment with system env vars
- Monitoring and alerts
- Team key management policy
