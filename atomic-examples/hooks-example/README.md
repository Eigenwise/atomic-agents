# AtomicAgent Hook System Example

This example demonstrates the powerful hook system integration in AtomicAgent, which leverages Instructor's hook system for comprehensive monitoring, error handling, and intelligent retry mechanisms.

## Features Demonstrated

- **🔍 Comprehensive Monitoring**: Track all aspects of agent execution
- **🛡️ Robust Error Handling**: Graceful handling of validation and completion errors
- **🔄 Intelligent Retry Patterns**: Implement smart retry logic based on error context
- **📊 Performance Metrics**: Monitor response times, success rates, and error patterns
- **🔧 Easy Debugging**: Detailed error information and execution flow visibility
- **⚡ Zero Overhead**: Hooks only execute when registered and enabled

## Getting Started

1. Clone the main Atomic Agents repository:
   ```bash
   git clone https://github.com/eigenwise/atomic-agents
   ```

2. Navigate to the hooks-example directory:
   ```bash
   cd atomic-agents/atomic-examples/hooks-example
   ```

3. Install the dependencies using uv:
   ```bash
   uv sync
   ```

4. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

5. Run the example:
   ```bash
   uv run python hooks_example/main.py
   ```

## What This Example Shows

The example demonstrates several key hook system patterns:

### Basic Hook Registration
- Simple parse error logging
- Completion monitoring and metrics collection

### Advanced Error Handling
- Comprehensive validation error analysis
- Intelligent retry mechanisms with backoff strategies
- Error isolation to prevent hook failures from disrupting execution

### Performance Monitoring
- Response time tracking
- Success rate calculation
- Error pattern analysis

### Real-World Scenarios
- Handling malformed responses
- Network timeouts and retry logic
- Model switching on repeated failures

## Key Benefits

This hook system implementation provides:

1. **Full Instructor Integration**: All Instructor hook events are supported
2. **Backward Compatibility**: Existing AtomicAgent code works unchanged
3. **Error Context**: Rich error information for intelligent decision making
4. **Performance Insights**: Detailed metrics for optimization
5. **Production Ready**: Robust error handling suitable for production use

## Hook Events Supported

- `parse:error` - Triggered on Pydantic validation failures
- `completion:kwargs` - Before API calls are made
- `completion:response` - After API responses are received
- `completion:error` - On API or network errors

## GitHub Issue Resolution

This example demonstrates the complete resolution of GitHub issue #173, showing how the AtomicAgent hook system enables:

- ✅ Parse error hooks triggering on validation failures
- ✅ Comprehensive error context for retry mechanisms
- ✅ Full Instructor hook event support
- ✅ 100% backward compatibility
- ✅ Robust error isolation

## Next Steps

After running this example, you can:

1. Experiment with different hook combinations
2. Implement custom retry strategies
3. Add your own monitoring and alerting logic
4. Explore integration with observability platforms
