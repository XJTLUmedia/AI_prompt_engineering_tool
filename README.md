# AI Prompt Engineering Tool

## 智能代码生成系统：FSM + RAG + 动态执行 + 运行时反馈 + 自动模块管理

这是一个基于状态机的AI代码生成系统，使用大型语言模型（LLM）自动生成、执行和优化Python代码。该系统集成了动态执行验证、自动错误修复和智能依赖管理功能。

## 功能特性

- **有限状态机（FSM）架构**：采用状态机设计，包括需求分析、接口设计、核心实现、动态验证、运行时修复等状态
- **动态代码执行**：在安全沙箱环境中执行生成的代码，自动捕获输出和异常
- **智能依赖管理**：自动检测和安装缺失的Python模块
- **运行时错误修复**：基于执行结果自动修复代码错误
- **网络错误处理**：增强的网络超时检测和错误处理机制
- **备用搜索引擎支持**：当主要搜索引擎失败时，自动切换到备用搜索引擎
- **RAG（检索增强生成）**：结合上下文信息生成更准确的代码

## 系统架构

### 核心组件

1. **PythonREPLExecutor**: 动态代码执行器，在隔离命名空间中执行代码
   - 捕获标准输出/错误
   - 捕获异常和完整堆栈跟踪
   - 自动检测和安装缺失模块
   - 增强网络错误检测

2. **EnhancedCodeGenerationStateMachine**: 增强型代码生成状态机
   - 需求分析 (ANALYZE)
   - 接口设计 (DESIGN)
   - 核心实现 (IMPLEMENT)
   - 动态执行验证 (DYNAMIC_VALIDATE)
   - 运行时修复 (REFINE)
   - 人工介入 (ESCALATE)
   - 终止 (TERMINAL)

3. **状态管理**:
   - 温度调度器：根据状态和错误类型动态调整生成参数
   - 错误分类器：对生成的错误进行分类和预算管理
   - 上下文压缩器：管理对话和执行历史

### 状态转换流程

```
ANALYZE → DESIGN → IMPLEMENT → DYNAMIC_VALIDATE → TERMINAL (成功)
                                      ↓
                                   REFINE ← → DYNAMIC_VALIDATE
                                      ↓
                                   ESCALATE (人工介入)
```

## 安装要求

### Python环境
- Python 3.8+

### 依赖包
项目依赖于以下主要包（详见requirements.txt）：

```bash
pip install -r requirements.txt
```

## 使用方法

1. **环境设置**：
   ```bash
   # 创建虚拟环境
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或 venv\Scripts\activate  # Windows
   
   # 安装依赖
   pip install -r requirements.txt
   ```

2. **配置API密钥**：
   ```bash
   # 设置环境变量(支持deepseek, kimi api)
   export DEEPSEEK_API_KEY=your_api_key_here
   export MOONSHOT_API_KEY=your_api_key_here
   $env:DEEPSEEK_API_KEY =your_api_key_here
   $env:MOONSHOT_API_KEY =your_api_key_here
   # 或在代码中直接配置
   ```

3. **运行系统**：
   ```bash
   python Askquestion2.py
   ```

## 工作流程

1. **需求分析**：系统分析用户输入的需求，生成结构化设计文档
2. **接口设计**：基于设计文档生成Python接口定义
3. **核心实现**：实现接口的具体逻辑
4. **动态验证**：在沙箱环境中执行代码，验证功能正确性
5. **运行时修复**：根据执行结果自动修复错误
6. **人工介入**：当自动修复失败时提供人工编辑选项

## 网络错误处理

系统增强了网络错误检测机制：
- 检测超时、连接错误等网络异常
- 实现备用搜索引擎机制
- 为网络错误分配独立的错误预算
- 在API调用中实现重试机制

## 错误类型

- `SYNTAX`: 语法错误
- `RUNTIME`: 运行时错误
- `EMPTY_CODE`: 空代码错误
- `OUTPUT_MISMATCH`: 输出不匹配
- `API_ERROR`: API错误
- `IMPORT_ERROR`: 导入错误
- `NETWORK_TIMEOUT`: 网络超时错误

## 自动模块管理

系统能够：
- 静默安装缺失的Python包
- 缓存已安装的模块信息
- 自动处理依赖关系
- 在安装失败时提供用户选择

## 测试功能

- 自动生成基于函数签名的测试用例
- 执行正常和边界输入测试
- 验证输出匹配
- 性能监控

## 扩展性

该系统设计为高度可扩展：
- 支持多种LLM提供商（OpenAI, DeepSeek等）
- 插件化架构便于添加新功能
- 灵活的状态管理机制
- 可配置的错误预算系统

## 注意事项

- 代码在沙箱环境中执行，但仍需注意安全风险
- 网络依赖的代码需要API密钥和网络连接
- 某些代码可能需要特定的系统权限
- 建议在虚拟环境中运行以避免依赖冲突

## 许可证

请参阅项目中的LICENSE文件（如果存在）。

## 贡献

欢迎提交Issue和Pull Request来改进此工具。

## 作者

AI Prompt Engineering Tool 开发团队
