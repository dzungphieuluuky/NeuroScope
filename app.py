import streamlit as st
import torch
import torch.nn as nn
import torch.onnx
import importlib.util
import sys
import os
import shutil
import zipfile
import tempfile
import ast
import json
import inspect
import pandas as pd
import gc
import re
from collections import OrderedDict
from streamlit_agraph import agraph, Node, Edge, Config
from torchview import draw_graph
from pathlib import Path
import textwrap
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
from io import BytesIO

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="NeuroScope Pro: Industry Standard Debugger")

# --- CUSTOM CSS FOR BETTER CODE DISPLAY ---
st.markdown("""
<style>
    .code-container {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        overflow-x: auto;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.5;
        max-height: 500px;
        overflow-y: auto;
    }
    .module-detail-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stat-box {
        background: rgba(255,255,255,0.1);
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        backdrop-filter: blur(10px);
    }
    .tensor-info {
        background: #2d3748;
        color: #e2e8f0;
        padding: 12px;
        border-radius: 6px;
        margin: 6px 0;
        font-family: monospace;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        background-color: #f7fafc;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# --- UTILITY: PROJECT LOADING & AST ANALYSIS ---

class ProjectLoader:
    """
    Handles File/Zip uploads, flattens structure, and resolves all imports transparently.
    """

    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file
        self.temp_dir = tempfile.mkdtemp()
        self.flat_dir = os.path.join(self.temp_dir, "flattened")
        self.is_zip = uploaded_file.name.endswith('.zip')
        self.module_cache = {}
        self.import_map = {}  # Maps original imports to flattened modules

    def __enter__(self):
        # Extract files
        if self.is_zip:
            zip_path = os.path.join(self.temp_dir, "project.zip")
            with open(zip_path, "wb") as f:
                f.write(self.uploaded_file.getbuffer())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)
            if os.path.exists(os.path.join(self.temp_dir, "__MACOSX")):
                shutil.rmtree(os.path.join(self.temp_dir, "__MACOSX"))
        else:
            file_path = os.path.join(self.temp_dir, self.uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(self.uploaded_file.getbuffer())

        # Create flattened directory
        os.makedirs(self.flat_dir, exist_ok=True)

        # Flatten structure: copy all .py files to flat_dir and resolve imports
        self._flatten_and_resolve()

        if self.flat_dir not in sys.path:
            sys.path.insert(0, self.flat_dir)
        
        # Pre-import all modules
        self._preload_modules()
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.flat_dir in sys.path:
            sys.path.remove(self.flat_dir)
        # Clear imported modules
        for mod_name in list(self.module_cache.keys()):
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        shutil.rmtree(self.temp_dir)

    def _flatten_and_resolve(self):
        """
        Flatten project structure and resolve all imports.
        - Copies all .py files to flat_dir with name changes to avoid collisions
        - Rewrites imports to match flattened structure
        - Creates a transparent module resolution system
        """
        # Step 1: Collect all Python files with their original paths
        file_mapping = {}  # {new_flat_name: (original_path, content)}
        
        for root, _, files in os.walk(self.temp_dir):
            # Skip the flattened dir and zip file
            if "flattened" in root or "__MACOSX" in root:
                continue
            
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.temp_dir)
                    
                    # Read original content
                    try:
                        with open(full_path, "r", encoding='utf-8') as f:
                            content = f.read()
                    except:
                        continue
                    
                    # Create flattened name (e.g., src/models/vgg.py -> src_models_vgg.py)
                    flat_name = rel_path.replace(os.sep, '_').replace('.py', '.py')
                    flat_name = flat_name.lstrip('_')
                    
                    file_mapping[flat_name] = {
                        'original_path': rel_path,
                        'content': content
                    }
        
        # Step 2: Build import resolution map
        self._build_import_map(file_mapping)
        
        # Step 3: Rewrite imports and save flattened files
        for flat_name, file_info in file_mapping.items():
            content = file_info['content']
            
            # Rewrite imports
            content = self._rewrite_imports(content, flat_name)
            
            # Save to flattened directory
            flat_path = os.path.join(self.flat_dir, flat_name)
            with open(flat_path, 'w', encoding='utf-8') as f:
                f.write(content)

    def _build_import_map(self, file_mapping):
        """
        Build a map of all possible imports and their flattened equivalents.
        """
        for flat_name, file_info in file_mapping.items():
            original_path = file_info['original_path']
            # Convert path to module name
            module_name = original_path.replace(os.sep, '.').replace('.py', '')
            
            # Map both original and flattened names
            self.import_map[module_name] = flat_name.replace('.py', '')
            
            # Also map directory-based imports (e.g., 'src.models' -> file in src/models/)
            parts = original_path.replace(os.sep, '.').replace('.py', '').split('.')
            for i in range(len(parts)):
                partial = '.'.join(parts[:i+1])
                if partial not in self.import_map:
                    self.import_map[partial] = flat_name.replace('.py', '')

    def _rewrite_imports(self, content, flat_name):
        """
        Rewrite all imports in the content to use flattened module names.
        """
        lines = content.split('\n')
        rewritten_lines = []
        
        for line in lines:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            indent_str = ' ' * indent
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                rewritten_lines.append(line)
                continue
            
            # Handle: from X import Y
            if stripped.startswith('from ') and ' import ' in stripped:
                match = re.match(r'from\s+([\w.]+)\s+import\s+(.+)', stripped)
                if match:
                    module_name = match.group(1)
                    imports = match.group(2)
                    
                    # Don't rewrite torch imports
                    if module_name.startswith('torch'):
                        rewritten_lines.append(line)
                    else:
                        resolved = self._resolve_module_name(module_name)
                        if resolved and resolved != module_name:
                            rewritten_lines.append(indent_str + f'from {resolved} import {imports}')
                        else:
                            rewritten_lines.append(line)
                else:
                    rewritten_lines.append(line)
            
            # Handle: import X [as Y]
            elif stripped.startswith('import '):
                match = re.match(r'import\s+([\w.]+)(?:\s+as\s+(\w+))?', stripped)
                if match:
                    module_name = match.group(1)
                    alias = match.group(2)
                    
                    # Don't rewrite torch imports
                    if module_name.startswith('torch'):
                        rewritten_lines.append(line)
                    else:
                        resolved = self._resolve_module_name(module_name)
                        if resolved and resolved != module_name:
                            if alias:
                                rewritten_lines.append(indent_str + f'import {resolved} as {alias}')
                            else:
                                rewritten_lines.append(indent_str + f'import {resolved}')
                        else:
                            rewritten_lines.append(line)
                else:
                    rewritten_lines.append(line)
            
            else:
                rewritten_lines.append(line)
        
        return '\n'.join(rewritten_lines)

    def _resolve_module_name(self, module_name):
        """
        Resolve a module name using the import map.
        """
        # Direct match
        if module_name in self.import_map:
            return self.import_map[module_name]
        
        # Try parent modules
        parts = module_name.split('.')
        for i in range(len(parts), 0, -1):
            partial = '.'.join(parts[:i])
            if partial in self.import_map:
                return self.import_map[partial]
        
        # No match found
        return module_name

    def _preload_modules(self):
        """
        Pre-import all Python modules with enhanced error handling.
        Loads modules in dependency order to resolve cross-imports.
        """
        max_retries = 3
        loaded_modules = set()
        remaining_files = {}
        
        # First pass: collect all files
        for root, _, files in os.walk(self.flat_dir):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    module_name = file.replace('.py', '')
                    remaining_files[module_name] = full_path
        
        # Try to load modules with retries (handles circular dependencies)
        for attempt in range(max_retries):
            modules_to_remove = []
            
            for module_name, full_path in remaining_files.items():
                if module_name in loaded_modules:
                    modules_to_remove.append(module_name)
                    continue
                
                try:
                    with open(full_path, "r", encoding='utf-8') as f:
                        code = f.read()
                    
                    # Security scan
                    dangerous_patterns = ["os.system", "subprocess.call", "exec(", "eval(", "__import__"]
                    if any(pattern in code for pattern in dangerous_patterns):
                        modules_to_remove.append(module_name)
                        continue
                    
                    spec = importlib.util.spec_from_file_location(module_name, full_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        # Make module available before execution (handles circular deps)
                        sys.modules[module_name] = module
                        
                        try:
                            spec.loader.exec_module(module)
                            self.module_cache[module_name] = module
                            loaded_modules.add(module_name)
                            modules_to_remove.append(module_name)
                        except ImportError as ie:
                            # Re-try later (dependency not yet loaded)
                            if attempt < max_retries - 1:
                                pass  # Keep in remaining_files
                            else:
                                modules_to_remove.append(module_name)
                        except Exception as e:
                            # Other errors - skip this module
                            modules_to_remove.append(module_name)
                except Exception as e:
                    modules_to_remove.append(module_name)
            
            # Remove successfully loaded or failed modules
            for module_name in modules_to_remove:
                remaining_files.pop(module_name, None)
            
            # If no modules left, we're done
            if not remaining_files:
                break

    def get_py_files(self):
        """
        Retrieve list of Python files in the flattened directory.
        """
        py_files = []
        for root, _, files in os.walk(self.flat_dir):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    py_files.append(os.path.relpath(full_path, self.flat_dir))
        return py_files

class ArchitectureScanner:
    """
    Scans .py files for classes inheriting from nn.Module (handles unresolved imports).
    """

    @staticmethod
    def scan(root_dir, file_rel_paths):
        """
        Scan files for nn.Module subclasses with robust import resolution.
        """
        found_classes = []
        
        for f_path in file_rel_paths:
            full_path = os.path.join(root_dir, f_path)
            try:
                with open(full_path, "r", encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                # Extract imports to build context
                imports = {}
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports[alias.asname or alias.name] = alias.name
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            for alias in node.names:
                                imports[alias.asname or alias.name] = node.module
                
                # Look for classes
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        is_module = False
                        base_names = []
                        
                        # Check all base classes
                        for base in node.bases:
                            base_names.append(base)
                            
                            # Direct: class X(Module)
                            if isinstance(base, ast.Name) and base.id == 'Module':
                                is_module = True
                            
                            # Qualified: class X(nn.Module)
                            elif isinstance(base, ast.Attribute):
                                if base.attr == 'Module':
                                    is_module = True
                        
                        # If not found directly, check if base is imported from torch.nn
                        if not is_module:
                            for base in base_names:
                                if isinstance(base, ast.Name):
                                    base_id = base.id
                                    # Check if this is imported from torch.nn
                                    if base_id in imports:
                                        import_source = imports[base_id]
                                        if 'torch.nn' in import_source or import_source == 'nn':
                                            is_module = True
                                            break
                        
                        if is_module:
                            found_classes.append({
                                'name': node.name,
                                'file': f_path.replace('.py', ''),  # Flattened module name
                                'imports': list(imports.values()),
                                'line': node.lineno
                            })
            except Exception as e:
                pass
        
        return found_classes

def load_dynamic_class(root_dir, module_name, class_name):
    """
    Load a class dynamically from a flattened module.
    """
    file_path = os.path.join(root_dir, module_name + '.py')
    
    try:
        # Check if already loaded
        if module_name in sys.modules:
            return getattr(sys.modules[module_name], class_name)
        
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            raise ImportError(f"Cannot create module spec for {module_name}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        
        try:
            spec.loader.exec_module(module)
        except Exception as exec_error:
            st.warning(f"Module execution warning: {exec_error}")
        
        if not hasattr(module, class_name):
            available = [name for name in dir(module) if not name.startswith('_') and isinstance(getattr(module, name), type)]
            raise AttributeError(
                f"Class '{class_name}' not found in module '{module_name}'. "
                f"Available classes: {', '.join(available[:5])}"
            )
        
        return getattr(module, class_name)
    except Exception as e:
        raise ValueError(f"Failed to load class {class_name} from {module_name}: {e}")

# --- UTILITY: INTERACTIVE ARGUMENT PARSING ---

def generate_init_form(model_class):
    """Introspects __init__, generates Streamlit widgets, returns args dict."""
    sig = inspect.signature(model_class.__init__)
    params = sig.parameters
    
    args = {}
    st.markdown("##### 🛠 Model Configuration")
    
    cols = st.columns(2)
    i = 0
    
    for name, param in params.items():
        if name == 'self':
            continue
        
        col = cols[i % 2]
        i += 1
        
        default = param.default if param.default is not inspect.Parameter.empty else None
        label = f"`{name}`"
        
        # Type inference and widget generation
        if param.annotation is int or isinstance(default, int):
            val = default if isinstance(default, int) else 3  # Default to 3 for most int params
            args[name] = col.number_input(label, value=val, step=1, help=f"Type: int")
            
        elif param.annotation is float or isinstance(default, float):
            val = default if isinstance(default, float) else 0.5  # Default to 0.5 for float params
            args[name] = col.number_input(label, value=val, format="%.4f", help=f"Type: float")
            
        elif param.annotation is bool or isinstance(default, bool):
            val = default if isinstance(default, bool) else False
            args[name] = col.checkbox(label, value=val)
            
        else:
            # Provide sensible defaults based on parameter name
            if 'channel' in name.lower() or 'input' in name.lower():
                default_val = default if default is not None else 3
            elif 'class' in name.lower() or 'output' in name.lower():
                default_val = default if default is not None else 10
            elif 'embed' in name.lower() or 'hidden' in name.lower():
                default_val = default if default is not None else 512
            elif 'layer' in name.lower() or 'block' in name.lower():
                default_val = default if default is not None else 4
            else:
                default_val = default
            
            val = str(default_val) if default_val is not None else ""
            input_val = col.text_input(label, value=val, help="Type: String/JSON. Enter lists/dicts as JSON.")
            
            try:
                if (input_val.startswith('[') and input_val.endswith(']')) or \
                   (input_val.startswith('{') and input_val.endswith('}')):
                    args[name] = json.loads(input_val)
                elif input_val.lower() == 'none':
                    args[name] = None
                else:
                    args[name] = input_val
            except:
                args[name] = input_val
                
    return args

# --- ENHANCED TRACER WITH DETAILED INFORMATION ---

class DetailedTracer:
    """Enhanced tracer with Netron-style detailed information."""
    def __init__(self):
        self.data = OrderedDict()
        self.hooks = []
        self.module_source_map = {}

    def register(self, model):
        def hook(name, module):
            def fn(mod, inp, out):
                # Input processing
                in_shapes = []
                in_dtypes = []
                for x in inp:
                    if isinstance(x, torch.Tensor):
                        in_shapes.append(list(x.shape))
                        in_dtypes.append(str(x.dtype))
                
                # Output processing
                if isinstance(out, tuple):
                    out_shapes = [list(x.shape) for x in out if isinstance(x, torch.Tensor)]
                    out_dtypes = [str(x.dtype) for x in out if isinstance(x, torch.Tensor)]
                elif isinstance(out, torch.Tensor):
                    out_shapes = [list(out.shape)]
                    out_dtypes = [str(out.dtype)]
                else:
                    out_shapes = []
                    out_dtypes = []
                
                # Weight and bias statistics
                params_info = {}
                if hasattr(mod, 'weight') and mod.weight is not None:
                    w = mod.weight
                    params_info['weight'] = {
                        'shape': list(w.shape),
                        'dtype': str(w.dtype),
                        'mean': w.mean().item(),
                        'std': w.std().item(),
                        'min': w.min().item(),
                        'max': w.max().item(),
                        'numel': w.numel()
                    }
                
                if hasattr(mod, 'bias') and mod.bias is not None:
                    b = mod.bias
                    params_info['bias'] = {
                        'shape': list(b.shape),
                        'dtype': str(b.dtype),
                        'mean': b.mean().item(),
                        'std': b.std().item(),
                        'numel': b.numel()
                    }
                
                # Module attributes (like kernel_size, stride, etc.)
                module_attrs = {}
                for attr in ['kernel_size', 'stride', 'padding', 'dilation', 'groups', 
                             'in_features', 'out_features', 'in_channels', 'out_channels',
                             'num_heads', 'embed_dim', 'dropout', 'eps', 'momentum']:
                    if hasattr(mod, attr):
                        module_attrs[attr] = getattr(mod, attr)
                
                # Source code extraction
                try:
                    source = inspect.getsource(mod.__class__)
                    source_file = inspect.getsourcefile(mod.__class__)
                except:
                    source = "Source not available"
                    source_file = "Unknown"

                self.data[name] = {
                    'type': mod.__class__.__name__,
                    'module': mod.__class__.__module__,
                    'in_shapes': in_shapes,
                    'in_dtypes': in_dtypes,
                    'out_shapes': out_shapes,
                    'out_dtypes': out_dtypes,
                    'params_total': sum(p.numel() for p in mod.parameters()),
                    'params_trainable': sum(p.numel() for p in mod.parameters() if p.requires_grad),
                    'params_info': params_info,
                    'module_attrs': module_attrs,
                    'source': source,
                    'source_file': source_file
                }
            return fn

        for n, m in model.named_modules():
            if n == "":
                continue
            self.hooks.append(m.register_forward_hook(hook(n, m)))

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.data = OrderedDict()

# --- CODE FORMATTING UTILITIES ---

def format_source_code(source_code, highlight_lines=None):
    """Format source code with syntax highlighting and optional line highlighting."""
    try:
        # Clean up the source code
        lines = source_code.split('\n')
        # Remove common leading whitespace
        min_indent = min((len(line) - len(line.lstrip()) for line in lines if line.strip()), default=0)
        cleaned_lines = [line[min_indent:] if len(line) >= min_indent else line for line in lines]
        cleaned_code = '\n'.join(cleaned_lines)
        
        # Syntax highlight
        formatter = HtmlFormatter(style='monokai', noclasses=True)
        highlighted = highlight(cleaned_code, PythonLexer(), formatter)
        
        return f'<div class="code-container">{highlighted}</div>'
    except:
        return f'<div class="code-container"><pre>{source_code}</pre></div>'

def display_module_details(module_name, module_info):
    """Display detailed module information in Netron-style."""
    st.markdown(f'<div class="module-detail-card">', unsafe_allow_html=True)
    st.markdown(f"### 🔍 {module_name}")
    st.markdown(f"**Type:** `{module_info['type']}`")
    st.markdown(f"**Module:** `{module_info['module']}`")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create columns for organized display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Tensor Information")
        
        # Input tensors
        st.markdown('<div class="tensor-info">', unsafe_allow_html=True)
        st.markdown("**Inputs:**")
        for i, (shape, dtype) in enumerate(zip(module_info['in_shapes'], module_info['in_dtypes'])):
            st.markdown(f"- Input {i}: `{shape}` | `{dtype}`")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Output tensors
        st.markdown('<div class="tensor-info">', unsafe_allow_html=True)
        st.markdown("**Outputs:**")
        for i, (shape, dtype) in enumerate(zip(module_info['out_shapes'], module_info['out_dtypes'])):
            st.markdown(f"- Output {i}: `{shape}` | `{dtype}`")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### ⚙️ Module Attributes")
        if module_info['module_attrs']:
            attrs_df = pd.DataFrame([module_info['module_attrs']]).T
            attrs_df.columns = ['Value']
            st.dataframe(attrs_df, width='stretch')
        else:
            st.info("No special attributes")
    
    # Parameters section
    if module_info['params_info']:
        st.markdown("#### 🎯 Parameters")
        st.markdown(f"**Total Parameters:** `{module_info['params_total']:,}`")
        st.markdown(f"**Trainable Parameters:** `{module_info['params_trainable']:,}`")
        
        for param_name, param_stats in module_info['params_info'].items():
            with st.expander(f"📊 {param_name.upper()} Statistics"):
                stats_df = pd.DataFrame([param_stats]).T
                stats_df.columns = ['Value']
                st.dataframe(stats_df, width='stretch')
    
    # Source code
    st.markdown("#### 💻 Source Code")
    st.caption(f"File: `{module_info.get('source_file', 'Unknown')}`")
    formatted_code = format_source_code(module_info['source'])
    st.markdown(formatted_code, unsafe_allow_html=True)

# --- MAIN UI ---

def main():
    st.sidebar.title("🧠 NeuroScope Pro")
    st.sidebar.markdown("**Advanced PyTorch Model Debugger**")
    st.sidebar.markdown("Generate **Torchview** graphs & **ONNX** exports with full module resolution.")
    st.sidebar.divider()

    # 1. UPLOAD
    uploaded_file = st.sidebar.file_uploader("📂 Upload Project (.py / .zip)", type=["py", "zip"])

    if uploaded_file:
        # Use session state to persist results
        upload_key = uploaded_file.name
        if 'scan_results' not in st.session_state or st.session_state.get('last_upload') != upload_key:
            with st.spinner("📂 Loading and analyzing project..."):
                with ProjectLoader(uploaded_file) as loader:
                    files = loader.get_py_files()
                    
                    # Debug info
                    with st.sidebar.expander("🔧 Debug Info"):
                        st.write(f"**Files found:** {len(files)}")
                        st.write(f"**Files:** {files}")
                        st.write(f"**Modules loaded:** {len(loader.module_cache)}")
                        if loader.module_cache:
                            st.write(f"**Module names:** {list(loader.module_cache.keys())}")
                    
                    classes = ArchitectureScanner.scan(loader.flat_dir, files)
                    st.session_state.scan_results = classes
                    st.session_state.last_upload = upload_key
        
        found_classes = st.session_state.scan_results
        
        if not found_classes:
            st.error("❌ No `nn.Module` classes found in the uploaded files.")
            st.info("""
            **Troubleshooting:**
            - Ensure your model classes inherit from `torch.nn.Module` or `nn.Module`
            - Check that `import torch.nn as nn` or `from torch.nn import Module` is present
            - Make sure the class is at module level (not nested inside a function)
            - Example: `class MyModel(nn.Module):`
            """)
            st.stop()

        # 2. SELECTION
        cls_options = [f"{c['name']} ({c['file']})" for c in found_classes]
        selected_option = st.sidebar.selectbox("Select Model Class", cls_options)
        selected_cls_name = selected_option.split(" (")[0]
        selected_module_name = selected_option[selected_option.index("(") + 1:selected_option.rindex(")")]

        # Display model info
        selected_class = next(c for c in found_classes if c['name'] == selected_cls_name and c['file'] == selected_module_name)
        with st.sidebar.expander("📋 Model Info"):
            st.write(f"**Module:** `{selected_module_name}`")
            st.write(f"**Line:** {selected_class['line']}")
            st.write(f"**Dependencies:** {len(selected_class['imports'])}")
            if selected_class['imports']:
                st.write(f"**Imports:** {', '.join(selected_class['imports'][:5])}")

        # 3. DYNAMIC CONFIGURATION
        init_args = {}
        export_onnx = False
        opset_version = 14
        dynamic_axes = False
        shape_str = "1, 3, 224, 224"
        
        with ProjectLoader(uploaded_file) as loader:
            try:
                ModelClass = load_dynamic_class(loader.flat_dir, selected_module_name, selected_cls_name)
                
                with st.sidebar.form("config_form"):
                    init_args = generate_init_form(ModelClass)
                    
                    st.markdown("##### 🔌 Input Tensor Config")
                    shape_str = st.text_input("Input Shape (N, C, H, W)", "1, 3, 224, 224")
                    
                    # Advanced options
                    with st.expander("⚙️ Advanced Options"):
                        export_onnx = st.checkbox("Export ONNX", value=True)
                        opset_version = st.number_input(
                            "ONNX Opset Version", 
                            value=14, 
                            min_value=9, 
                            max_value=18,
                            help="Higher versions support newer operators but less hardware support"
                        )
                        dynamic_axes = st.checkbox("Enable Dynamic Axes", value=False)
                    
                    load_btn = st.form_submit_button("🚀 Compile & Trace", use_container_width=True)
            except Exception as e:
                st.sidebar.error(f"Error inspecting class: {e}")
                st.stop()

        # 4. EXECUTION
        if load_btn:
            with st.spinner("🔄 Compiling Model & Tracing Graph..."):
                with ProjectLoader(uploaded_file) as loader:
                    try:
                        # Instantiate
                        ModelClass = load_dynamic_class(loader.flat_dir, selected_module_name, selected_cls_name)
                        model = ModelClass(**init_args)
                        model.eval()
                        
                        # Parse Shape with validation
                        try:
                            shape = [int(x.strip()) for x in shape_str.split(',')]
                            if len(shape) < 1 or any(s <= 0 for s in shape):
                                st.error("❌ Invalid input shape. All dimensions must be positive integers.")
                                st.stop()
                            dummy_input = torch.randn(*shape)
                        except ValueError:
                            st.error("❌ Invalid input shape format. Use comma-separated integers (e.g., '1, 3, 224, 224')")
                            st.stop()
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # A. Interactive Trace (Custom Hooks)
                        status_text.text("Registering hooks...")
                        progress_bar.progress(25)
                        
                        tracer = DetailedTracer()
                        tracer.register(model)
                        
                        try:
                            status_text.text("Running forward pass...")
                            progress_bar.progress(50)
                            
                            with torch.no_grad():
                                output = model(dummy_input)
                            st.session_state.trace_data = tracer.data
                            st.session_state.trace_status = "success"
                            st.session_state.model_output = output
                        except Exception as e:
                            st.error(f"Forward Pass Failed: {e}")
                            st.session_state.trace_data = tracer.data
                            st.session_state.trace_status = "failed"

                        # B. Torchview Trace
                        status_text.text("Generating Torchview graph...")
                        progress_bar.progress(75)
                        
                        try:
                            graph_view = draw_graph(model, input_size=tuple(shape), expand_nested=True)
                            st.session_state.graph_dot = graph_view.visual_graph
                        except Exception as e:
                            st.warning(f"Torchview tracing failed: {e}")
                            st.session_state.graph_dot = None

                        # C. ONNX Export
                        if export_onnx:
                            status_text.text("Exporting to ONNX...")
                            progress_bar.progress(90)
                            
                            st.session_state.onnx_bytes = None
                            try:
                                onnx_buffer = BytesIO()
                                
                                # Use opset 18 or higher for best compatibility
                                effective_opset = max(int(opset_version), 18)
                                
                                # Build export parameters
                                export_kwargs = {
                                    # 'input_names': ['input'],
                                    # 'output_names': ['output'],
                                }
                                
                                # Add dynamic axes if enabled
                                if dynamic_axes:
                                    export_kwargs['dynamic_shapes'] = {
                                        'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}
                                    }
                                
                                # Export to ONNX
                                torch.onnx.export(
                                    model,
                                    (dummy_input,),
                                    "my_model.onnx",
                                    input_names=['input'],
                                    dynamo=True,
                                )
                                
                                # Get bytes from buffer
                                st.session_state.onnx_bytes = onnx_buffer.getvalue()
                                
                            except ImportError as ie:
                                st.error("ONNX Export failed: Missing required package. Run `pip install torch onnx`.")
                            except Exception as e:
                                st.warning(f"ONNX Export failed: {str(e)[:200]}")

                        progress_bar.progress(100)
                        status_text.text("✅ Complete!")
                        
                        # Model summary
                        total_params = sum(p.numel() for p in model.parameters())
                        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        st.session_state.model_summary = {
                            'total_params': total_params,
                            'trainable_params': trainable_params,
                            'model_size_mb': total_params * 4 / (1024 * 1024)
                        }
                        
                        # Cleanup
                        tracer.clear()
                        gc.collect()

                    except Exception as e:
                        st.error(f"Critical Error: {e}")
                        st.exception(e)

    # --- VISUALIZATION TABS ---
    if 'trace_data' in st.session_state:
        # Display model summary at top
        if 'model_summary' in st.session_state:
            summary = st.session_state.model_summary
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Parameters", f"{summary['total_params']:,}")
            col2.metric("Trainable Parameters", f"{summary['trainable_params']:,}")
            col3.metric("Model Size", f"{summary['model_size_mb']} MB")
            col4.metric("Trace Status", "✅ Success" if st.session_state.trace_status == "success" else "⚠️ Failed")
        
        st.divider()
        
        t1, t2, t3, t4 = st.tabs(["🔎 Interactive Debugger", "🕸️ Torchview Graph", "📊 Module Analysis", "📤 Export"])
        
        # TAB 1: INTERACTIVE DEBUGGER
        with t1:
            data = st.session_state.trace_data
            
            if not data:
                st.warning("No trace data available. Model may have failed during forward pass.")
            else:
                col_graph, col_info = st.columns([2, 1])
                
                with col_graph:
                    st.markdown("### 🗺️ Computational Graph")
                    
                    # Build Node Graph
                    nodes, edges = [], []
                    keys = list(data.keys())
                    
                    for i, (k, v) in enumerate(data.items()):
                        color = "#ffcccb" if st.session_state.trace_status == "failed" and i == len(data)-1 else "#90EE90"
                        
                        label = f"{k.split('.')[-1]}\n{v['type']}\n{v['out_shapes'][0] if v['out_shapes'] else 'N/A'}"
                        nodes.append(Node(id=k, label=label, shape="box", color=color, size=25))
                        if i > 0:
                            edges.append(Edge(source=keys[i-1], target=k))
                    
                    config = Config(
                        width="100%", 
                        height=700, 
                        directed=True, 
                        nodeHighlightBehavior=True,
                        highlightColor="#667eea"
                    )
                    sel_node = agraph(nodes, edges, config)

                with col_info:
                    st.markdown("### 📌 Quick Info")
                    st.caption("Click on a node in the graph to see details")
                    
                    if sel_node and sel_node in data:
                        info = data[sel_node]
                        st.markdown(f"**Selected:** `{sel_node}`")
                        st.markdown(f"**Type:** `{info['type']}`")
                        st.markdown(f"**Parameters:** `{info['params_total']:,}`")
                        
                        if info['out_shapes']:
                            st.markdown(f"**Output Shape:** `{info['out_shapes'][0]}`")
                        
                        if st.button("📋 View Full Details", use_container_width=True):
                            st.session_state.selected_module = sel_node
                    else:
                        st.info("Select a module from the graph")

        # TAB 2: TORCHVIEW
        with t2:
            st.markdown("### 🌐 High-Fidelity Structural Graph")
            st.caption("Powered by `torchview`. Shows strict data dependencies and nested module structure.")
            
            if 'graph_dot' in st.session_state and st.session_state.graph_dot:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.graphviz_chart(st.session_state.graph_dot)
                
                with col2:
                    st.markdown("#### 🎯 Graph Legend")
                    st.markdown("""
                    - **Rectangles**: Operations/Layers
                    - **Arrows**: Data flow
                    - **Labels**: Tensor shapes
                    - **Colors**: Operation types
                    """)
                    
                    if st.button("🔍 Inspect Module", use_container_width=True):
                        st.info("Click on nodes in the Interactive Debugger tab for detailed inspection")
            else:
                st.warning("Graph not available. Model may be incompatible with Torchview.")

        # TAB 3: MODULE ANALYSIS
        with t3:
            st.markdown("### 🔬 Detailed Module Analysis")
            data = st.session_state.trace_data
            
            if 'selected_module' in st.session_state:
                module_name = st.session_state.selected_module
                if module_name in data:
                    display_module_details(module_name, data[module_name])
            else:
                st.info("👈 Select a module from the Interactive Debugger tab to view detailed analysis")
                
                # Show all modules list
                st.markdown("#### 📚 All Modules")
                module_list = list(data.keys())
                
                # Create searchable/filterable module list
                col1, col2 = st.columns([2, 1])
                with col1:
                    search_term = st.text_input("🔍 Search modules", placeholder="e.g., conv, linear, attention")
                with col2:
                    filter_type = st.selectbox("Filter by type", ["All"] + list(set([data[m]['type'] for m in module_list])))
                
                # Filter modules
                filtered_modules = module_list
                if search_term:
                    filtered_modules = [m for m in filtered_modules if search_term.lower() in m.lower()]
                if filter_type != "All":
                    filtered_modules = [m for m in filtered_modules if data[m]['type'] == filter_type]
                
                st.caption(f"Showing {len(filtered_modules)} of {len(module_list)} modules")
                
                # Display modules in a grid
                for i in range(0, len(filtered_modules), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        if i + j < len(filtered_modules):
                            mod_name = filtered_modules[i + j]
                            mod_info = data[mod_name]
                            
                            with col:
                                with st.container(border=True):
                                    st.markdown(f"**{mod_name}**")
                                    st.caption(f"Type: `{mod_info['type']}`")
                                    
                                    # Mini stats
                                    subcol1, subcol2 = st.columns(2)
                                    subcol1.metric("Params", f"{mod_info['params_total']:,}", label_visibility="collapsed")
                                    if mod_info['out_shapes']:
                                        subcol2.metric("Output", str(mod_info['out_shapes'][0]), label_visibility="collapsed")
                                    
                                    if st.button(f"🔍 Inspect", key=f"inspect_{mod_name}", use_container_width=True):
                                        st.session_state.selected_module = mod_name
                                        st.rerun()

        # TAB 4: EXPORT
        with t4:
            st.markdown("### 📦 Model Export & Documentation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 ONNX Export")
                if st.session_state.get('onnx_bytes'):
                    st.success("✅ ONNX model generated successfully!")
                    st.markdown("""
                    **Next Steps:**
                    1. Download the `.onnx` file below
                    2. Open it in [Netron.app](https://netron.app)
                    3. Explore the computational graph interactively
                    
                    **Benefits:**
                    - Cross-platform model visualization
                    - Detailed operator-level inspection
                    - Compatible with deployment frameworks
                    """)
                    
                    # ONNX file info
                    onnx_size = len(st.session_state.onnx_bytes) / (1024 * 1024)
                    st.info(f"📊 ONNX File Size: {onnx_size} MB")
                    
                    st.download_button(
                        label="⬇️ Download ONNX Model",
                        data=st.session_state.onnx_bytes,
                        file_name="model_neuroscope.onnx",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                    
                    # Additional info
                    with st.expander("ℹ️ About ONNX Format"):
                        st.markdown("""
                        **ONNX (Open Neural Network Exchange)** is an open format for representing machine learning models.
                        
                        - Supports model deployment across frameworks
                        - Enables hardware acceleration optimization
                        - Industry-standard for model interchange
                        - Compatible with TensorRT, OpenVINO, CoreML, and more
                        """)
                else:
                    st.error("❌ ONNX export failed or was disabled")
                    st.markdown("""
                    **Possible reasons:**
                    - ONNX export was disabled in configuration
                    - Model contains unsupported operations
                    - Dynamic control flow detected
                    
                    **Solutions:**
                    - Enable ONNX export in the sidebar
                    - Simplify model architecture
                    - Check error logs above
                    """)
            
            with col2:
                st.markdown("#### 📄 Model Report")
                
                # Generate comprehensive report
                if st.button("📝 Generate Full Report", use_container_width=True):
                    data = st.session_state.trace_data
                    report_lines = []
                    report_lines.append("# Model Architecture Report\n")
                    report_lines.append(f"**Generated:** {pd.Timestamp.now()}\n")
                    report_lines.append(f"**Model Class:** {selected_cls_name}\n")
                    report_lines.append(f"**Source Module:** {selected_module_name}\n\n")
                    
                    if 'model_summary' in st.session_state:
                        summary = st.session_state.model_summary
                        report_lines.append("## Model Summary\n\n")
                        report_lines.append(f"| Metric | Value |\n")
                        report_lines.append(f"|--------|-------|\n")
                        report_lines.append(f"| Total Parameters | {summary['total_params']:,} |\n")
                        report_lines.append(f"| Trainable Parameters | {summary['trainable_params']:,} |\n")
                        report_lines.append(f"| Model Size | {summary['model_size_mb']} MB |\n\n")
                    
                    report_lines.append("## Layer-by-Layer Analysis\n\n")
                    
                    for idx, (mod_name, mod_info) in enumerate(data.items(), 1):
                        report_lines.append(f"### {idx}. {mod_name}\n\n")
                        report_lines.append(f"**Type:** `{mod_info['type']}`  \n")
                        report_lines.append(f"**Module Path:** `{mod_info['module']}`  \n")
                        report_lines.append(f"**Total Parameters:** {mod_info['params_total']:,}  \n")
                        report_lines.append(f"**Trainable Parameters:** {mod_info['params_trainable']:,}  \n\n")
                        
                        if mod_info['in_shapes']:
                            report_lines.append(f"**Input Shapes:**\n")
                            for i, (shape, dtype) in enumerate(zip(mod_info['in_shapes'], mod_info['in_dtypes'])):
                                report_lines.append(f"- Input {i}: `{shape}` ({dtype})\n")
                            report_lines.append("\n")
                        
                        if mod_info['out_shapes']:
                            report_lines.append(f"**Output Shapes:**\n")
                            for i, (shape, dtype) in enumerate(zip(mod_info['out_shapes'], mod_info['out_dtypes'])):
                                report_lines.append(f"- Output {i}: `{shape}` ({dtype})\n")
                            report_lines.append("\n")
                        
                        if mod_info['module_attrs']:
                            report_lines.append(f"**Attributes:**\n")
                            for attr, val in mod_info['module_attrs'].items():
                                report_lines.append(f"- {attr}: `{val}`\n")
                            report_lines.append("\n")
                        
                        if mod_info['params_info']:
                            report_lines.append(f"**Parameter Details:**\n\n")
                            for param_name, param_stats in mod_info['params_info'].items():
                                report_lines.append(f"*{param_name.capitalize()}:*\n")
                                report_lines.append(f"```\n")
                                for stat_key, stat_val in param_stats.items():
                                    report_lines.append(f"{stat_key}: {stat_val}\n")
                                report_lines.append(f"```\n\n")
                        
                        report_lines.append("---\n\n")
                    
                    report = "".join(report_lines)
                    
                    st.download_button(
                        label="⬇️ Download Report (Markdown)",
                        data=report,
                        file_name=f"model_report_{selected_cls_name}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                    
                    st.success("✅ Report generated successfully!")
                
                # Quick stats preview
                st.markdown("#### 📊 Quick Statistics")
                
                data = st.session_state.trace_data
                if data:
                    # Calculate statistics - keep as integers, not formatted strings
                    total_layers = len(data)
                    param_layers = sum(1 for m in data.values() if m['params_total'] > 0)
                    total_params = sum(m['params_total'] for m in data.values())
                    
                    stats_df = pd.DataFrame({
                        'Metric': ['Total Layers', 'Parametric Layers', 'Total Parameters'],
                        'Value': [total_layers, param_layers, total_params]
                    })
                    
                    st.dataframe(stats_df, hide_index=True, width='stretch')
            
            # Additional export options
            st.divider()
            st.markdown("#### 🔧 Additional Export Options")
            
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.markdown("**📊 Parameters CSV**")
                if st.button("Generate CSV", key="export_csv", use_container_width=True):
                    data = st.session_state.trace_data
                    params_data = []
                    for mod_name, mod_info in data.items():
                        params_data.append({
                            'Module': mod_name,
                            'Type': mod_info['type'],
                            'Total_Parameters': mod_info['params_total'],
                            'Trainable_Parameters': mod_info['params_trainable'],
                            'Output_Shape': str(mod_info['out_shapes'][0]) if mod_info['out_shapes'] else 'N/A'
                        })
                    
                    df = pd.DataFrame(params_data)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv,
                        file_name=f"model_parameters_{selected_cls_name}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col4:
                st.markdown("**📋 Layer Summary**")
                if st.button("Generate Summary", key="layer_summary", use_container_width=True):
                    data = st.session_state.trace_data
                    # Create a summary of layer types
                    layer_types = {}
                    for mod_info in data.values():
                        layer_type = mod_info['type']
                        if layer_type not in layer_types:
                            layer_types[layer_type] = {'count': 0, 'params': 0}
                        layer_types[layer_type]['count'] += 1
                        layer_types[layer_type]['params'] += mod_info['params_total']
                    
                    summary_lines = ["# Layer Type Summary\n\n"]
                    summary_lines.append("| Layer Type | Count | Total Parameters |\n")
                    summary_lines.append("|------------|-------|------------------|\n")
                    
                    for layer_type, stats in sorted(layer_types.items(), key=lambda x: x[1]['params'], reverse=True):
                        summary_lines.append(f"| {layer_type} | {stats['count']} | {stats['params']:,} |\n")
                    
                    summary = "".join(summary_lines)
                    
                    st.download_button(
                        label="⬇️ Download Summary",
                        data=summary,
                        file_name=f"layer_summary_{selected_cls_name}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
            
            with col5:
                st.markdown("**🔄 Session**")
                if st.button("Reset Session", key="reset_session", use_container_width=True):
                    # Clear all session state
                    keys_to_delete = list(st.session_state.keys())
                    for key in keys_to_delete:
                        del st.session_state[key]
                    st.success("Session reset! Refresh page to start over.")
                    st.rerun()
    
    else:
        # Welcome screen
        st.markdown("""
        # 🧠 Welcome to NeuroScope Pro
        
        ### The Ultimate PyTorch Model Debugger
        
        **Features:**
        - 📂 **Multi-file Project Support** - Upload entire projects as ZIP files with automatic dependency resolution
        - 🔍 **Interactive Graph Visualization** - Click and explore your model's computational graph in real-time
        - 📊 **Detailed Module Analysis** - Netron-style detailed information for every layer including tensor shapes and statistics
        - 💻 **Source Code Inspection** - View and analyze module implementations with syntax highlighting
        - 📤 **ONNX Export** - Export your models for deployment and external visualization tools
        - 📈 **Performance Metrics** - Track parameters, shapes, and data flow through your network
        - 🎯 **Smart Module Detection** - Automatically discovers all nn.Module classes in your project
        
        ### Getting Started
        
        1. **Upload** your PyTorch project (`.py` file or `.zip` archive) using the sidebar
        2. **Select** the model class you want to analyze from the detected classes
        3. **Configure** initialization parameters and input tensor shapes interactively
        4. **Click** "Compile & Trace" to analyze your model architecture
        5. **Explore** the interactive visualizations and detailed module information across tabs
        
        ### Supported Features
        
        - ✅ Custom `nn.Module` classes with complex architectures
        - ✅ Multi-file projects with cross-module dependencies and imports
        - ✅ Dynamic argument configuration with type inference
        - ✅ Comprehensive parameter statistics (mean, std, min, max)
        - ✅ Interactive computational graph with node selection
        - ✅ Syntax-highlighted source code viewing with smart formatting
        - ✅ ONNX export with configurable opset versions (9-18)
        - ✅ Detailed model reports and documentation generation
        - ✅ CSV export for parameters and layer analysis
        
        ### Advanced Capabilities
        
        - **Cross-file dependency resolution** - Automatically handles imports between project files
        - **Weight statistics** - Detailed analysis of weights and biases for each layer
        - **Tensor flow tracking** - Monitor shapes and dtypes through the entire network
        - **Module attributes** - Inspect kernel sizes, strides, padding, and other configuration
        - **Searchable module list** - Filter and find specific layers quickly
        - **Export flexibility** - Multiple output formats for different use cases
        
        ---
        
        **👈 Get started by uploading a file in the sidebar**
        """)
        
        # Example code snippet
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("📝 Example: Simple CNN"):
                st.code('''
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 56 * 56, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
                ''', language='python')
        with col2:
            with st.expander("📝 Example: Multi-file Project"):
                st.code('''
# File: modules.py
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# File: model.py
from modules import ConvBlock

class CustomNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = ConvBlock(3, 64)
        self.block2 = ConvBlock(64, 128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
                ''', language='python')
        
        # Tips section
        st.markdown("---")
        st.markdown("### 💡 Pro Tips")
        
        tip_col1, tip_col2, tip_col3 = st.columns(3)
        
        with tip_col1:
            st.info("""
            **📦 ZIP Projects**
            
            Upload entire project folders as ZIP files. NeuroScope will automatically:
            - Detect all Python files
            - Resolve imports
            - Find all nn.Module classes
            """)
        
        with tip_col2:
            st.info("""
            **🔍 Module Inspection**
            
            Click on any node in the Interactive Debugger to:
            - View detailed statistics
            - Inspect source code
            - Analyze tensor shapes
            """)
        
        with tip_col3:
            st.info("""
            **📊 Export Options**
            
            Multiple export formats:
            - ONNX for deployment
            - Markdown reports
            - CSV for analysis
            - Layer summaries
            """)

if __name__ == "__main__":
    main()