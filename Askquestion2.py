#!/usr/bin/env python3
"""
智能代码生成系统：FSM + RAG + 动态执行 + 运行时反馈 + 自动模块管理
修复版：增强网络错误检测、添加备用搜索引擎支持
"""

import builtins
import os
import sys
import json
import time
import tempfile
import subprocess
import ast
import re
import io
import contextlib
import traceback
from openai import OpenAI
from enum import Enum, auto
import importlib.util
import socket  # 新增：用于网络超时检测

from typing import List, Dict, Optional, Tuple, Callable, Any, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# ==================== 核心数据结构与配置 ====================

class State(Enum):
    ANALYZE = auto()      # 需求分析
    DESIGN = auto()       # 接口设计
    IMPLEMENT = auto()    # 核心实现
    DYNAMIC_VALIDATE = auto()  # 动态执行验证
    REFINE = auto()       # 运行时修复
    ESCALATE = auto()     # 人工介入
    TERMINAL = auto()     # 终止

class ErrorType(Enum):
    SYNTAX = "syntax"
    RUNTIME = "runtime"
    EMPTY_CODE = "empty_code"
    OUTPUT_MISMATCH = "output_mismatch"
    API_ERROR = "api_error"
    UNKNOWN = "unknown"
    IMPORT_ERROR = "import_error"
    NETWORK_TIMEOUT = "network_timeout"  # 新增：网络超时错误

@dataclass
class ExecutionResult:
    """动态执行结果"""
    code: str
    stdout: str
    stderr: str
    exception: Optional[Exception] = None
    exception_traceback: str = ""
    execution_time: float = 0.0
    success: bool = False
    output_match: bool = False
    installed_modules: List[str] = field(default_factory=list)
    network_errors: List[str] = field(default_factory=list)  # 新增：记录网络错误

# ==================== 增强的 REPL 执行器 ====================

class PythonREPLExecutor:
    """
    动态执行器：
    1. 在隔离命名空间中执行代码
    2. 捕获 stdout/stderr
    3. 捕获异常和完整堆栈
    4. 自动检测和安装缺失模块
    5. 增强网络错误检测
    """
    
    def __init__(self, timeout: int = 3000):
        self.timeout = timeout
        self.execution_history: List[ExecutionResult] = []
        self.auto_install = True
        self.installed_modules_cache = set()
    
    def _extract_imports(self, code: str) -> Set[str]:
        """AST 分析：提取所有导入的模块名"""
        try:
            tree = ast.parse(code)
            imports = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        imports.add(module_name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        imports.add(module_name)
            
            return imports
        except:
            return set()
    
    def _is_module_installed(self, module_name: str) -> bool:
        """检查模块是否已安装"""
        try:
            if module_name in self.installed_modules_cache:
                return True
                
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                self.installed_modules_cache.add(module_name)
                return True
            return False
        except:
            return False
    
    def _install_module(self, module_name: str) -> bool:
        """通过 pip 安装模块"""
        try:
            print(f"[模块安装] 正在安装 {module_name}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "--quiet", module_name
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print(f"[模块安装] ✅ {module_name} 安装成功")
                self.installed_modules_cache.add(module_name)
                return True
            else:
                print(f"[模块安装] ❌ {module_name} 安装失败: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print(f"[模块安装] ❌ {module_name} 安装超时")
            return False
        except Exception as e:
            print(f"[模块安装] ❌ {module_name} 安装出错: {e}")
            return False
    
    def _check_and_install_modules(self, code: str) -> Tuple[bool, List[str]]:
        """检查并安装缺失的模块"""
        imports = self._extract_imports(code)
        
        builtin_modules = set(sys.builtin_module_names)
        modules_to_check = imports - builtin_modules
        
        if not modules_to_check:
            return True, []
        
        print(f"[模块检查] 发现外部模块: {', '.join(modules_to_check)}")
        
        installed = []
        all_success = True
        
        for module_name in modules_to_check:
            if not self._is_module_installed(module_name):
                print(f"[模块检查] {module_name} 未安装")
                
                if self.auto_install:
                    success = self._install_module(module_name)
                    if success:
                        installed.append(module_name)
                    else:
                        all_success = False
                else:
                    choice = input(f"是否安装模块 '{module_name}'? (y/n): ").strip().lower()
                    if choice == 'y':
                        if self._install_module(module_name):
                            installed.append(module_name)
                        else:
                            all_success = False
                    else:
                        print(f"[模块检查] 跳过安装 {module_name}")
                        all_success = False
            else:
                print(f"[模块检查] ✅ {module_name} 已安装")
        
        return all_success, installed
    
    def _is_network_error(self, exception: Exception) -> bool:
        """检测是否为网络相关异常"""
        if not exception:
            return False
        
        error_msg = str(exception).lower()
        network_keywords = [
            'timeout', 'timed out', 'connection', 'connect',
            'max retries exceeded', 'refused', 'unreachable',
            'network', 'socket', 'urllib', 'requests'
        ]
        
        return any(keyword in error_msg for keyword in network_keywords)
    
    def execute(self, code: str) -> ExecutionResult:
        """执行代码并捕获所有输出和异常"""
        print("[执行前检查] 正在分析代码依赖...")
        install_success, installed_modules = self._check_and_install_modules(code)
        
        if not install_success:
            print("[警告] 部分模块安装失败，可能导致执行错误")
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        network_errors = []
        
        start_time = time.time()
        
        try:
            compiled_code = compile(code, '<dynamic>', 'exec')
            sandbox = {'__builtins__': builtins}
            
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                exec(compiled_code, sandbox)
            
            execution_time = time.time() - start_time
            
            # 检查输出中是否包含网络错误信息
            output_content = stdout_capture.getvalue() + stderr_capture.getvalue()
            if 'Network error' in output_content or 'timed out' in output_content:
                network_errors.append(f"检测到网络异常: {output_content[:200]}")
            
            # 自动提取函数并测试
            functions = self._extract_function_signatures(code)
            test_results = []
            
            for func_info in functions[:3]:
                test_cases = self._generate_test_inputs(func_info)
                for test_case in test_cases[:2]:
                    try:
                        func = sandbox.get(func_info['name'])
                        if callable(func):
                            result = func(**test_case['inputs'])
                            test_results.append({
                                'test': test_case['description'],
                                'result': str(result)[:100],
                                'passed': True
                            })
                    except Exception as e:
                        if self._is_network_error(e):
                            network_errors.append(f"网络错误: {str(e)[:100]}")
                        test_results.append({
                            'test': test_case['description'],
                            'error': str(e),
                            'passed': False
                        })
            
            # 修复：有网络错误时判定为失败
            success = len(network_errors) == 0 and all(r.get('passed', False) for r in test_results)
            
            return ExecutionResult(
                code=code,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                exception=None,
                execution_time=execution_time,
                success=success,
                output_match=len(test_results) > 0 and all(r.get('passed', False) for r in test_results),
                installed_modules=installed_modules,
                network_errors=network_errors
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # 检测网络异常
            is_network_error = self._is_network_error(e)
            if is_network_error:
                network_errors.append(f"执行期网络错误: {str(e)[:200]}")
            
            return ExecutionResult(
                code=code,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                exception=e,
                exception_traceback=traceback.format_exc(),
                execution_time=execution_time,
                success=False,
                installed_modules=installed_modules,
                network_errors=network_errors
            )
    
    def _extract_function_signatures(self, code: str) -> List[Dict[str, Any]]:
        """自动提取函数签名"""
        try:
            tree = ast.parse(code)
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    args = []
                    defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + node.args.defaults
                    
                    for i, arg in enumerate(node.args.args):
                        default = defaults[i] if i < len(defaults) and defaults[i] is not None else None
                        args.append({
                            'name': arg.arg,
                            'type': ast.unparse(arg.annotation) if arg.annotation else 'Any',
                            'default': ast.unparse(default) if default else None
                        })
                    
                    functions.append({
                        'name': node.name,
                        'args': args,
                        'has_return': any(isinstance(n, ast.Return) for n in ast.walk(node))
                    })
            
            return functions
        except:
            return []
    
    def _generate_test_inputs(self, func_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于函数签名生成测试输入"""
        test_cases = []
        func_name = func_info['name']
        args = func_info['args']
        
        # 正常用例
        normal_case = {}
        for arg in args:
            arg_type = arg['type']
            if arg_type == 'int' or 'int' in arg_type:
                normal_case[arg['name']] = 42 if 'size' in arg['name'] or 'length' in arg['name'] else 100
            elif arg_type == 'str' or 'str' in arg_type:
                if 'name' in arg['name']:
                    normal_case[arg['name']] = "test_user"
                elif 'path' in arg['name']:
                    normal_case[arg['name']] = "/tmp/test"
                else:
                    normal_case[arg['name']] = "sample_string"
            elif arg_type == 'list' or 'List' in arg_type:
                normal_case[arg['name']] = [1, 2, 3]
            elif arg_type == 'dict' or 'Dict' in arg_type:
                normal_case[arg['name']] = {"key": "value"}
            else:
                normal_case[arg['name']] = None
        
        if normal_case:
            test_cases.append({
                'description': f"正常输入: {func_name}",
                'inputs': normal_case,
                'expect_output': True
            })
        
        # 边界用例
        if args:
            edge_case = {}
            for arg in args:
                arg_type = arg['type']
                if arg_type == 'int':
                    edge_case[arg['name']] = 0
                elif arg_type == 'str':
                    edge_case[arg['name']] = "" if arg.get('default') is None else arg['default']
                elif arg_type == 'list':
                    edge_case[arg['name']] = []
                else:
                    edge_case[arg['name']] = None
            
            test_cases.append({
                'description': f"边界输入: {func_name}",
                'inputs': edge_case,
                'expect_output': True
            })
        
        return test_cases

# ==================== 增强状态机 ====================

class EnhancedCodeGenerationStateMachine:
    """增强版：集成动态执行与运行时反馈"""
    
    def __init__(self, client: OpenAI, provider: str = "deepseek", model: str = None):
        self.client = client
        self.provider = provider
        self.model = model or ("deepseek-reasoner" if provider == "deepseek" else 
                              "kimi-k2-thinking" if provider == "kimi" else "openai")
        
        self.state = State.ANALYZE
        self.states = {
            State.ANALYZE: self.analyze_requirements,
            State.DESIGN: self.design_interface,
            State.IMPLEMENT: self.implement_core,
            State.DYNAMIC_VALIDATE: self.dynamic_validate,
            State.REFINE: self.refine_with_runtime_feedback,
            State.ESCALATE: self.request_human_intervention
        }
        
        self.context = ContextCompressor()
        self.temperature_scheduler = TemperatureScheduler()
        self.error_classifier = ErrorClassifier()
        self.repl_executor = PythonREPLExecutor()
        
        # 状态数据
        self.prd: str = ""
        self.design_doc: str = ""
        self.interface_code: str = ""
        self.implementation_code: str = ""
        self.execution_result: Optional[ExecutionResult] = None
        self.all_errors: List[str] = []
        self.generated_test_cases: List[Dict[str, Any]] = []
        
        # 错误预算
        self.error_budget = {
            ErrorType.SYNTAX: 3,
            ErrorType.RUNTIME: 3,
            ErrorType.EMPTY_CODE: 2,
            ErrorType.OUTPUT_MISMATCH: 2,
            ErrorType.IMPORT_ERROR: 2,
            ErrorType.NETWORK_TIMEOUT: 3  # 新增网络超时预算
        }
        
        # 状态转换图
        self.transitions = {
            State.ANALYZE: [
                Transition(State.DESIGN, lambda: bool(self.design_doc), priority=1)
            ],
            State.DESIGN: [
                Transition(State.IMPLEMENT, lambda: bool(self.interface_code), priority=1),
                Transition(State.ESCALATE, lambda: self._get_retry_count() > 2, priority=0)
            ],
            State.IMPLEMENT: [
                Transition(State.DYNAMIC_VALIDATE, lambda: True, priority=2),
            ],
            State.DYNAMIC_VALIDATE: [
                Transition(State.TERMINAL, lambda: self._execution_success(), priority=2),
                Transition(State.REFINE, lambda: self._needs_refinement(), priority=1),
                Transition(State.ESCALATE, lambda: not self._retry_available(), priority=0)
            ],
            State.REFINE: [
                Transition(State.DYNAMIC_VALIDATE, lambda: self._retry_available(), priority=2),
                Transition(State.ESCALATE, lambda: not self._retry_available(), priority=0)
            ]
        }
    
    def _get_retry_count(self) -> int:
        return self.temperature_scheduler.retry_count
    
    def _has_critical_error(self) -> bool:
        return any(ErrorType.EMPTY_CODE.value in e or ErrorType.SYNTAX.value in e 
                  for e in self.all_errors)
    
    def _execution_success(self) -> bool:
        """修复：网络错误也视为失败"""
        if not self.execution_result:
            return False
        
        # 有网络错误即视为失败
        if self.execution_result.network_errors:
            return False
        
        return self.execution_result.success

    def _needs_refinement(self) -> bool:
        """明确需要修复的条件"""
        return bool(self.all_errors) or bool(self.execution_result and self.execution_result.network_errors)
    
    def _retry_available(self) -> bool:
        """检查是否还有重试预算"""
        for error in self.all_errors:
            err_type = self.error_classifier.categorize(error)
            if self.error_budget.get(err_type, 0) > 0:
                return True
        
        # 检查网络错误预算
        if self.execution_result and self.execution_result.network_errors:
            return self.error_budget.get(ErrorType.NETWORK_TIMEOUT, 0) > 0
        
        return self._get_retry_count() < 5
    
    async def call_api(self, prompt: str, max_tokens: int = 8000, 
                       temperature: Optional[float] = None) -> str:
        """调用LLM API"""
        try:
            temp = temperature or self.temperature_scheduler.get_temperature(
                self.state,
                self.error_classifier.categorize(self.all_errors[-1]) if self.all_errors else None
            )
            
            print(f"\n[API调用] {self.provider}.{self.model} | 状态: {self.state.name} | 温度: {temp:.2f}")
            print(f"[提示词] {len(prompt)} 字符")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": 
                     "You are an expert Python developer. MUST output ONLY executable code. "
                     "NEVER include explanations, markdown, or apologies. "
                     "CRITICAL: Code must be self-contained and runnable."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=max_tokens,
                stream=False,
            )
            
            result = response.choices[0].message.content
            if not result:
                raise ValueError("API返回空响应")
            
            self.context.append("assistant", result)
            return result
            
        except Exception as e:
            print(f"[API错误] {type(e).__name__}: {e}")
            raise
    
    def _extract_code(self, response: str) -> str:
        """提取代码"""
        code = re.sub(r'```python\n|```\n|```', '', response).strip()
        
        if len(code) < 100:
            raise ValueError(f"提取的代码过短（{len(code)}字符），可能提取失败")
        
        return code
    
    async def analyze_requirements(self, prd: str) -> str:
        print(f"\n{'='*70}")
        print(f"[状态] ANALYZE: 需求分析")
        print(f"{'='*70}")
        
        prompt = f"""Analyze this requirement and output a structured design doc:
REQUIREMENT: {prd}

OUTPUT FORMAT:
- Feature decomposition (bullet points)
- Core interface signatures (Python function defs with type hints)
- Algorithm/approach justification
- Error handling strategy

CRITICAL: Focus on what to implement, not how."""
        
        self.design_doc = await self.call_api(prompt, max_tokens=8000)
        return self.design_doc
    
    async def design_interface(self, _) -> str:
        print(f"\n{'='*70}")
        print(f"[状态] DESIGN: 接口设计")
        print(f"{'='*70}")
        
        prompt = f"""Based on this design doc, generate ONLY Python interface definitions:
DESIGN: {self.design_doc}

OUTPUT:
- Class/function signatures with docstrings
- Type annotations
- TODO comments for implementation
- NO actual logic

EXAMPLE:
def process(data: List[int]) -> Dict[str, int]:
    '''Process integer list'''
    TODO: Implement
    pass"""
        
        response = await self.call_api(prompt, max_tokens=8000)
        self.interface_code = self._extract_code(response)
        return self.interface_code
    
    async def implement_core(self, _) -> str:
        print(f"\n{'='*70}")
        print(f"[状态] IMPLEMENT: 核心实现")
        print(f"{'='*70}")
        
        # 增强提示：明确要求添加备用搜索引擎
        prompt = f"""Implement the complete, runnable Python code:
DESIGN: {self.design_doc}
INTERFACE: {self.interface_code}

REQUIREMENTS:
- Replace ALL TODO with real implementation
- Include self-contained test logic if `if __name__ == "__main__":`
- Handle errors gracefully
- CRITICAL: For network operations, implement fallback mechanisms
- Add Baidu search as backup when Google fails
- Must be executable directly"""
        
        self.temperature_scheduler.reset_retry()
        response = await self.call_api(prompt, max_tokens=8000)
        self.implementation_code = self._extract_code(response)
        
        # 语法快速检查
        try:
            compile(self.implementation_code, '<check>', 'exec')
        except SyntaxError as e:
            self.all_errors = [f"{ErrorType.SYNTAX.value}: {e}"]
            self.temperature_scheduler.increment_retry()
        
        return self.implementation_code
    
    async def dynamic_validate(self, _) -> ExecutionResult:
        """状态4: 动态执行验证"""
        print(f"\n{'='*70}")
        print(f"[状态] DYNAMIC_VALIDATE: 动态执行与测试")
        print(f"{'='*70}")
        
        print(f"[动态执行] 正在执行代码... (timeout={self.repl_executor.timeout}s)")
        result = self.repl_executor.execute(self.implementation_code)
        self.execution_result = result
        
        # 修复：正确显示网络错误
        if result.network_errors:
            print(f"[网络错误] {'❌ 检测到网络问题'}")
            for err in result.network_errors:
                print(f"  - {err}")
        
        print(f"[执行结果] {'✅ 成功' if result.success else '❌ 失败'} | 耗时: {result.execution_time:.2f}s")
        
        if result.installed_modules:
            print(f"[模块安装] 已安装: {', '.join(result.installed_modules)}")
        if result.stdout:
            print(f"\n[标准输出]\n{result.stdout[:500]}...")
        if result.stderr:
            print(f"\n[标准错误]\n{result.stderr[:500]}...")
        if result.exception:
            print(f"\n[异常]\n{result.exception}\n{result.exception_traceback[:500]}...")
        
        # 生成反馈提示
        if not result.success or result.network_errors:
            self.all_errors = []
            if result.exception:
                if "ImportError" in str(type(result.exception)) or "ModuleNotFoundError" in str(type(result.exception)):
                    self.all_errors.append(f"{ErrorType.IMPORT_ERROR.value}: {result.exception}")
                elif self.repl_executor._is_network_error(result.exception):
                    self.all_errors.append(f"{ErrorType.NETWORK_TIMEOUT.value}: {result.exception}")
                else:
                    self.all_errors.append(f"{ErrorType.RUNTIME.value}: {result.exception}")
            
            if result.network_errors:
                self.all_errors.extend(result.network_errors)
            
            if not result.output_match:
                self.all_errors.append(f"{ErrorType.OUTPUT_MISMATCH.value}: 测试用例未通过")
            
            # 消耗错误预算
            for error in self.all_errors:
                err_type = self.error_classifier.categorize(error)
                if err_type in self.error_budget:
                    self.error_budget[err_type] -= 1
            
            self.context.append("runtime_error", result.exception_traceback or result.stderr or "\n".join(result.network_errors))
            self.temperature_scheduler.increment_retry()
        
        return result
    
    async def refine_with_runtime_feedback(self, _) -> str:
        """状态5: 基于运行时错误修复"""
        print(f"\n{'='*70}")
        print(f"[状态] REFINE: 运行时修复")
        print(f"{'='*70}")
        
        error_info = ""
        if self.execution_result:
            if self.execution_result.exception:
                error_info = f"EXCEPTION: {self.execution_result.exception}\nTRACEBACK:\n{self.execution_result.exception_traceback}"
            elif self.execution_result.network_errors:
                error_info = f"NETWORK ERRORS:\n" + "\n".join(self.execution_result.network_errors)
            elif self.execution_result.stderr:
                error_info = f"STDERR: {self.execution_result.stderr}"
            else:
                error_info = f"OUTPUT MISMATCH: 测试未通过\nSTDOUT: {self.execution_result.stdout}"
        
        prompt = f"""Fix the runtime errors in this code:
CODE:
{self.implementation_code}

ERROR:
{error_info}

REQUIREMENTS:
- Fix the root cause, not just symptoms
- For network errors, implement better error handling or use fallback
- Preserve function signatures
- Add error handling if needed
- Output ONLY the corrected code

FIXED CODE:"""
        
        response = await self.call_api(prompt, max_tokens=8000)
        new_code = self._extract_code(response)
        
        # 验证修复后的代码
        new_result = self.repl_executor.execute(new_code)
        if new_result.success:
            self.implementation_code = new_code
            self.execution_result = new_result
        else:
            print("[修复失败] 修复后代码仍无法执行")
        
        return self.implementation_code
    
    async def request_human_intervention(self, _) -> str:
        print(f"\n{'='*70}")
        print(f"[状态] ESCALATE: 人工介入")
        print(f"{'='*70}")
        
        print("\n系统无法自动修复，请选择:")
        print("1. 手动编辑代码 (M)")
        print("2. 提供修复提示 (P)")
        print("3. 查看执行历史 (H)")
        print("4. 退出 (Q)")
        
        choice = input("> ").strip().upper()
        
        if choice == 'M':
            fp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
            fp.write(self.implementation_code)
            fp.close()
            
            print(f"\n编辑保存: {fp.name}")
            input("完成后按 Enter...")
            
            with open(fp.name, 'r') as f:
                self.implementation_code = f.read()
            
            self.state = State.DYNAMIC_VALIDATE
            return "manual_fix"
        
        elif choice == 'P':
            hint = input("修复提示: ")
            self.context.append("human_hint", hint)
            self.state = State.REFINE
            return "hint_provided"
        
        elif choice == 'H':
            print("\n执行历史:")
            for i, h in enumerate(self.repl_executor.execution_history[-3:], 1):
                print(f"{i}. {'成功' if h.success else '失败'} | {h.code[:50]}...")
        
        else:
            sys.exit(0)
    
    def _decide_next_state(self) -> State:
        """状态机决策核心"""
        transitions = self.transitions.get(self.state, [])
        sorted_transitions = sorted(transitions, key=lambda t: t.priority, reverse=True)
        
        for trans in sorted_transitions:
            if trans.condition():
                return trans.to_state
        
        return self.state
    
    async def run(self, prd: str) -> str:
        """主循环"""
        self.prd = prd
        self.context.append("user", prd)
        
        print(f"\n{'='*70}")
        print(f"🚀 启动动态代码生成状态机 ({self.provider})")
        print(f"需求: {prd}...")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        while self.state != State.TERMINAL:
            print(f"\n[状态] {self.state.name}")
            
            handler = self.states[self.state]
            await handler(self.prd)
            
            next_state = self._decide_next_state()
            if next_state != self.state:
                print(f"[转换] {self.state.name} → {next_state.name}")
                self.state = next_state
            
            if time.time() - start_time > 900:
                print("[警告] 超时退出")
                break
        
        # 最终执行报告
        if self.execution_result:
            print(f"\n{'='*70}")
            print("最终执行报告")
            print(f"{'='*70}")
            print(f"执行结果: {'✅ 成功' if self.execution_result.success else '❌ 失败'}")
            print(f"执行时间: {self.execution_result.execution_time:.2f}s")
            if self.execution_result.installed_modules:
                print(f"已安装模块: {', '.join(self.execution_result.installed_modules)}")
            if self.execution_result.network_errors:
                print(f"网络错误: {len(self.execution_result.network_errors)} 个")
            if self.execution_result.stdout:
                print(f"输出:\n{self.execution_result.stdout[:500]}")
        
        return self.implementation_code


# ==================== 工具函数与类 ====================

@dataclass
class ContextItem:
    role: str
    content: str
    timestamp: float
    is_decision: bool = False

class ContextCompressor:
    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.context_window: List[ContextItem] = []
        self.decision_keywords = ["选择", "因此", "权衡", "最终决定", "instead of",
                                  "rather than", "架构", "设计", "decision", "choose"]

    def append(self, role: str, content: str):
        is_decision = any(kw in content.lower() for kw in self.decision_keywords)
        item = ContextItem(role=role, content=content if is_decision else self._summarize(content),
                           timestamp=time.time(), is_decision=is_decision)
        self.context_window.append(item)
        self._enforce_limit()

    def _summarize(self, text: str, max_len: int = 150) -> str:
        return text if len(text) <= max_len else text[:max_len] + "... [truncated]"

    def _enforce_limit(self):
        while sum(len(it.content) // 4 for it in self.context_window) > self.max_tokens:
            for idx, it in enumerate(self.context_window):
                if not it.is_decision:
                    self.context_window.pop(idx)
                    break
            else:
                self.context_window.pop(0)

    def get_relevant_context(self, target_state: State, top_k: int = 3) -> List[ContextItem]:
        keywords = {
            State.ANALYZE: ["需求", "功能", "requirement"],
            State.DESIGN: ["interface", "API", "接口"],
            State.IMPLEMENT: ["algorithm", "code", "实现"],
            State.DYNAMIC_VALIDATE: ["test", "验证"],
            State.REFINE: ["optimize", "fix", "优化"]
        }.get(target_state, [])

        def score(it: ContextItem) -> float:
            s = 100 if it.is_decision else 0
            s += sum(10 for kw in keywords if kw in it.content.lower())
            s += it.timestamp / 1000
            return s

        return sorted(self.context_window, key=score, reverse=True)[:top_k]

class Transition:
    def __init__(self, to_state: State, condition: Callable[[], bool], 
                 validator: Optional[Callable[[str], bool]] = None, priority: int = 0):
        self.to_state = to_state
        self.condition = condition
        self.validator = validator
        self.priority = priority

class TemperatureScheduler:
    def __init__(self,
                 base_temps: Optional[Dict[State, float]] = None,
                 error_multiplier: Optional[Dict[ErrorType, float]] = None,
                 retry_factor: float = 0.02,
                 max_retry_penalty: float = 0.15,
                 max_temp: float = 0.35):
        
        default_temps = {
            State.ANALYZE: 0.1,
            State.DESIGN: 0.3,
            State.IMPLEMENT: 0.3,
            State.DYNAMIC_VALIDATE: 0.05,
            State.REFINE: 0.08
        }
        self.base_temps = base_temps if base_temps is not None else default_temps
        
        default_multipliers = {
            ErrorType.SYNTAX: 0.5,
            ErrorType.RUNTIME: 0.7,
            ErrorType.EMPTY_CODE: 0.3,
            ErrorType.OUTPUT_MISMATCH: 0.6,
            ErrorType.API_ERROR: 0.9,
            ErrorType.UNKNOWN: 1.0,
            ErrorType.IMPORT_ERROR: 0.4,
            ErrorType.NETWORK_TIMEOUT: 0.5  # 新增网络超时乘数
        }
        self.error_multiplier = error_multiplier if error_multiplier is not None else default_multipliers
        
        self.retry_factor = retry_factor
        self.max_retry_penalty = max_retry_penalty
        self.max_temp = max_temp
        self.retry_count = 0
    
    def get_temperature(self, state: State, error_type: Optional[ErrorType] = None) -> float:
        temp = self.base_temps.get(state, 0.1)
        
        if error_type:
            multiplier = self.error_multiplier.get(error_type, 1.0)
            temp *= multiplier
        
        retry_penalty = self.retry_factor * self.retry_count
        if retry_penalty > self.max_retry_penalty:
            retry_penalty = self.max_retry_penalty
        
        temp += retry_penalty
        temp = max(0.01, min(temp, self.max_temp))
        
        return temp
    
    def reset_retry(self):
        self.retry_count = 0
    
    def increment_retry(self):
        self.retry_count += 1

class ErrorClassifier:
    @staticmethod
    def categorize(error: str) -> ErrorType:
        if not error:
            return ErrorType.UNKNOWN
        
        e = error.lower()
        if "empty_code" in e:
            return ErrorType.EMPTY_CODE
        if "syntaxerror" in e or "indentation" in e:
            return ErrorType.SYNTAX
        if "runtime" in e or "exception" in e:
            return ErrorType.RUNTIME
        if "output" in e and "mismatch" in e:
            return ErrorType.OUTPUT_MISMATCH
        if "import" in e or "module" in e:
            return ErrorType.IMPORT_ERROR
        if "timeout" in e or "timed out" in e or "connection" in e:  # 新增网络超时检测
            return ErrorType.NETWORK_TIMEOUT
        if "network" in e:
            return ErrorType.NETWORK_TIMEOUT
        
        return ErrorType.UNKNOWN

# ==================== 主入口 ====================

def setup_llm_client() -> Tuple[OpenAI, str]:
    """初始化 LLM 客户端"""
    
    def test_client(client, provider, model, test_message="test"):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": test_message}],
                max_tokens=5,
                timeout=300
            )
            return True, None
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)[:100]}"

    if key := os.environ.get("DEEPSEEK_API_KEY"):
        try:
            # 修复：移除尾随空格
            client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
            success, error = test_client(client, "deepseek", "deepseek-reasoner")
            if success:
                print("[连接] ✅ DeepSeek API")
                return client, "deepseek"
            else:
                print(f"[连接] ❌ DeepSeek API: {error}")
        except Exception as e:
            print(f"[连接] ❌ DeepSeek API 初始化失败: {e}")
    
    if key := os.environ.get("MOONSHOT_API_KEY"):
        try:
            # 修复：移除尾随空格
            client = OpenAI(api_key=key, base_url="https://api.moonshot.cn/v1")
            success, error = test_client(client, "kimi", "kimi-k2-thinking")
            if success:
                print("[连接] ✅ Kimi (Moonshot) API")
                return client, "kimi"
            else:
                print(f"[连接] ❌ Kimi API: {error}")
        except Exception as e:
            print(f"[连接] ❌ Kimi API 初始化失败: {e}")
    
    # print("[连接] 使用 Pollination AI (免费)")
    # client = OpenAI(api_key="pollination", base_url="https://text.pollinations.ai/openai")
    # return client, "pollination"
    
        # 无可用 API，提示用户并退出
    print("\n" + "="*70)
    print("❌ 未配置任何可用的 LLM API")
    print("="*70)
    print("请设置以下环境变量之一：")
    print("  export DEEPSEEK_API_KEY='your-deepseek-api-key'")
    print("  export MOONSHOT_API_KEY='your-moonshot-api-key'")
    print("Windows: ")
    print("$env:DEEPSEEK_API_KEY =your_api_key_here")
    print("$env:MOONSHOT_API_KEY =your_api_key_here")
    print("\n获取 API Key：")
    print("  - DeepSeek: https://platform.deepseek.com")
    print("  - Kimi: https://platform.moonshot.cn")
    print("="*70)
    
    sys.exit(1)
    
def main():
    """交互式主入口"""
    print("="*70)
    print("动态代码生成系统 (FSM + RAG + REPL执行 + 运行时反馈 + 自动模块管理)")
    print("="*70)
    
    client, provider = setup_llm_client()
    fsm = EnhancedCodeGenerationStateMachine(client, provider=provider)
    
    print("\n请输入需求描述 (或 'demo' 使用示例):")
    user_input = input("> ").strip()
    
    if user_input.lower() == "demo":
        prd = "请生成一个数据质量监控python代码，要求可以验证个人身份信息，信息表有10个常见字段，如身份证号码，地 址，个人名字，性别，年龄，学历，工资，银行账号等"
        print(f"\n使用示例需求: {prd}")
    elif not user_input:
        print("输入不能为空!")
        return
    else:
        prd = user_input
    
    try:
        import asyncio
        final_code = asyncio.run(fsm.run(prd))
        
        print("\n" + "="*70)
        print("最终生成代码:")
        print("="*70 + "\n")
        print(final_code)
        
        # 最终手动执行选项
        print("\n" + "="*70)
        if input("是否立即执行生成的代码? (y/n): ").lower() == 'y':
            print("\n[手动执行] 正在运行...")
            executor = PythonREPLExecutor()
            result = executor.execute(final_code)
            print(f"[执行结果] {'✅ 成功' if result.success else '❌ 失败'}")
            if result.installed_modules:
                print(f"[模块安装] 已安装: {', '.join(result.installed_modules)}")
            if result.network_errors:
                print(f"[网络错误] {len(result.network_errors)} 个")
            if result.stdout:
                print(f"\n[输出]\n{result.stdout}")
            if result.exception:
                print(f"\n[异常]\n{result.exception}")
        
        # 保存选项
        print("\n" + "="*70)
        if input("是否保存到文件? (y/n): ").lower() == 'y':
            filename = f"generated_{int(time.time())}.py"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(final_code)
            print(f"✅ 已保存: {os.path.abspath(filename)}")
        
    except KeyboardInterrupt:
        print("\n\n用户中断，退出...")
        sys.exit(0)
    except Exception as e:
        print(f"\n[致命错误] {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()