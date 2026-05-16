import os
import re
from pathlib import Path

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Skip files that shouldn't be touched
    if "logger.hpp" in filepath or "logger.cpp" in filepath:
        return

    # Add include if not present and we match std::cout/std::cerr
    if 'std::cout' in content or 'std::cerr' in content:
        if '#include <utils/logging/logger.hpp>' not in content:
            # Find the last include
            lines = content.split('\n')
            last_include = -1
            for i, line in enumerate(lines):
                if line.startswith('#include'):
                    last_include = i
            
            if last_include != -1:
                lines.insert(last_include + 1, '#include <utils/logging/logger.hpp>')
                content = '\n'.join(lines)

    # Now replace std::cout << ... ; with TREX_INFO(...) ;
    # We use a regex that matches std::cout followed by anything up to a semicolon.
    # Non-greedy .*? to avoid capturing multiple statements.
    
    # We need to handle std::cout << foo << bar; -> TREX_INFO(foo << bar);
    # First, let's substitute std::cerr to TREX_WARN
    def repl_info(match):
        inner = match.group(1)
        # remove trailing whitespace
        return f"TREX_INFO({inner.strip()});"

    def repl_warn(match):
        inner = match.group(1)
        return f"TREX_WARN({inner.strip()});"

    # Match std::cout << (stuff) ;
    # Specifically: std::cout\s*<<(.*?);  where .*? matches across newlines!
    content = re.sub(r'std::cout\s*<<\s*(.*?);', repl_info, content, flags=re.DOTALL)
    content = re.sub(r'std::cerr\s*<<\s*(.*?);', repl_warn, content, flags=re.DOTALL)

    with open(filepath, 'w') as f:
        f.write(content)

src_dir = Path("src")
for root, _, files in os.walk(src_dir):
    for file in files:
        if file.endswith(('.cpp', '.hpp')):
            process_file(os.path.join(root, file))

print("Rewrite done.")
