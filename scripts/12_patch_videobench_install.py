import importlib.util
import py_compile
import re
from pathlib import Path


def patch_prompt_dict(root: Path) -> bool:
    path = root / "prompt_dict.py"
    text = path.read_text(encoding="utf-8")
    if '"video-text consistency": overall_consistency_prompt' in text:
        return False
    needle = '    "overall_consistency": overall_consistency_prompt,\n'
    replacement = (
        '    "overall_consistency": overall_consistency_prompt,\n'
        '    "video-text consistency": overall_consistency_prompt,\n'
    )
    if needle not in text:
        raise RuntimeError(f"Unable to patch prompt_dict.py at {path}")
    path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    return True


def patch_init(root: Path) -> bool:
    path = root / "__init__.py"
    text = path.read_text(encoding="utf-8")
    if 'prompt_pairs = []' in text and 'candidate_paths = []' in text:
        return False
    pattern = r"if mode == 'custom_nonstatic':.*?\n\s*elif mode == 'custom_static':"
    replacement = """if mode == 'custom_nonstatic':
            self.check_dimension_requires_extra_info(dimension_list)
            actual_dimensions = set(dimension_mapping[dim] for dim in dimension_list)

            prompt_pairs = []
            if isinstance(prompt_list, str):
                prompt_pairs.append((str(prompt_list), str(prompt_list)))
            elif isinstance(prompt_list, dict):
                for prompt_key, prompt_text in prompt_list.items():
                    prompt_pairs.append((str(prompt_key), str(prompt_text)))
            else:
                for item in prompt_list:
                    prompt_pairs.append((str(item), str(item)))
            print(f\"Prompts to process: {[pair[1] for pair in prompt_pairs]}\")

            for prompt_key, prompt_text in prompt_pairs:
                videos_for_prompt = {}

                for actual_dim in actual_dimensions:
                    candidate_paths = []
                    dimension_path = os.path.join(videos_path, actual_dim)
                    if os.path.isdir(dimension_path):
                        candidate_paths.append(dimension_path)
                    if os.path.isdir(videos_path):
                        candidate_paths.append(videos_path)

                    seen_candidates = set()
                    for candidate_path in candidate_paths:
                        if candidate_path in seen_candidates:
                            continue
                        seen_candidates.add(candidate_path)
                        available_models = [name for name in os.listdir(candidate_path) if os.path.isdir(os.path.join(candidate_path, name))]
                        for model_name in available_models:
                            model_path = os.path.join(candidate_path, model_name)
                            if not os.path.isdir(model_path):
                                continue
                            for video_name in os.listdir(model_path):
                                if Path(video_name).suffix.lower() not in ['.mp4', '.gif', '.jpg', '.png']:
                                    continue
                                extracted_prompt = get_prompt_from_filename(video_name)
                                prompt_match = extracted_prompt == prompt_key
                                if (not prompt_match) and prompt_key.isdigit():
                                    zero_pad = f\"{int(prompt_key):04d}\"
                                    prompt_match = extracted_prompt == zero_pad or extracted_prompt.startswith(zero_pad + '_')
                                if not prompt_match:
                                    continue
                                video_path = os.path.join(model_path, video_name)
                                videos_for_prompt[model_name] = video_path.replace('\\', '/')
                        if videos_for_prompt:
                            break

                if videos_for_prompt:
                    cur_full_info_list.append({
                        \"prompt_en\": prompt_text,
                        \"dimension\": dimension_list,
                        \"videos\": videos_for_prompt
                    })

        elif mode == 'custom_static':"""
    new_text, count = re.subn(pattern, replacement, text, flags=re.S)
    if count != 1:
        raise RuntimeError(f"Unable to patch __init__.py at {path}; count={count}")
    path.write_text(new_text, encoding="utf-8")
    return True


def patch_video_text_alignment(root: Path) -> bool:
    path = root / "VideoTextAlignment.py"
    text = path.read_text(encoding="utf-8")
    changed = False

    old = """    for i in l1:
        data = dataset[i]
        modelmessage = f\"{len(data['frames'][modelname])} frames from {modelname}.\"
            
        agents = [Agent('Assistant-one', logger, prompt, config), 
                    Agent('Assistant-two', logger, prompt, config)]
        host = Host('Host', logger, prompt, config, modelname, modelmessage, agents)
        
        for agent in agents:
            agent.video_prompt = data['prompt']
        host.video_prompt = data['prompt']
        host.frames = data['frames']
        
        score[i] = {}
        history[i] = {}
        score[i]['prompt_en'] = data['prompt']

        # 动态获取模型列表
        available_models = list(data['frames'].keys())
        models_to_process = models if models else available_models
        
        for modelname in models_to_process:
"""
    new = """    for i in l1:
        data = dataset[i]

        score[i] = {}
        history[i] = {}
        score[i]['prompt_en'] = data['prompt']

        # 动态获取模型列表
        available_models = list(data['frames'].keys())
        models_to_process = models if models else available_models

        for modelname in models_to_process:
            if modelname not in data['frames']:
                continue

            modelmessage = f\"{len(data['frames'][modelname])} frames from {modelname}.\"
            agents = [Agent('Assistant-one', logger, prompt, config),
                      Agent('Assistant-two', logger, prompt, config)]
            host = Host('Host', logger, prompt, config, modelname, modelmessage, agents)

            for agent in agents:
                agent.video_prompt = data['prompt']
            host.video_prompt = data['prompt']
            host.frames = data['frames']

"""
    if old in text:
        text = text.replace(old, new, 1)
        changed = True

    replacements = {
        "self.api_key = self.config['GPT4o_mini_API_KEY']": "self.api_key = self.config.get('GPT4o_mini_API_KEY', self.config.get('OPENAI_API_KEY', 'local-videobench'))",
        "self.base_url = self.config['GPT4o_mini_BASE_URL']": "self.base_url = self.config.get('GPT4o_mini_BASE_URL', self.config.get('OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1'))",
        'self.model = "gpt-4o-mini"': "self.model = self.config.get('GPT4o_mini_MODEL', self.config.get('OPENAI_MINI_MODEL', self.config.get('OPENAI_MODEL', 'gpt-4o-mini')))",
        "self.api_key = self.config['GPT4o_API_KEY']": "self.api_key = self.config.get('GPT4o_API_KEY', self.config.get('OPENAI_API_KEY', 'local-videobench'))",
        "self.base_url = self.config['GPT4o_BASE_URL']": "self.base_url = self.config.get('GPT4o_BASE_URL', self.config.get('OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1'))",
        'self.model = "gpt-4o-2024-08-06"': "self.model = self.config.get('GPT4o_MODEL', self.config.get('OPENAI_MODEL', 'gpt-4o-2024-08-06'))",
    }
    for old_line, new_line in replacements.items():
        if old_line in text:
            text = text.replace(old_line, new_line)
            changed = True

    old_extract = """def extract_content_from_result(final_result):
    text = str(final_result or "")

    start_index = text.find("Updated Video Description")
    if start_index != -1:
        all_content = text[start_index + len("Updated Video Description"):].strip()
        eval_result_index = all_content.find("Evaluation Result")
        if eval_result_index != -1:
            remaining_content = all_content[eval_result_index + len("Evaluation Result"):].strip()
            because_index = remaining_content.lower().find("because")
            target = remaining_content if because_index == -1 else remaining_content[:because_index].strip()
            matches = re.findall(r"(?<!\\d)(10|[0-9])(?!\\d)", target)
            if matches:
                return int(matches[-1])

    patterns = [
        r"(?i)evaluation\\s*result[^0-9]{0,20}(10|[0-9])",
        r"(?i)score[^0-9]{0,20}(10|[0-9])",
        r"(?i)rating[^0-9]{0,20}(10|[0-9])",
        r"(?i)overall[^0-9]{0,20}(10|[0-9])",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))

    matches = re.findall(r"(?<!\\d)(10|[0-9])(?!\\d)", text)
    if matches:
        return int(matches[-1])

    print("No information found.")
    return 0
"""
    new_extract = """def extract_content_from_result(final_result):
    text = str(final_result or "")

    def parse_segment(segment):
        candidate = str(segment or "").strip()
        if not candidate:
            return None
        if candidate in {"1", "2", "3"}:
            return int(candidate)

        patterns = [
            r"(?im)^\\s*\\[?evaluation\\s*result\\]?\\s*[:\\-]?\\s*([123])\\b",
            r"(?im)^\\s*\\[?score\\]?\\s*[:\\-]?\\s*([123])\\b",
            r"(?im)^\\s*\\[?rating\\]?\\s*[:\\-]?\\s*([123])\\b",
            r"(?im)^\\s*\\[?overall\\]?\\s*[:\\-]?\\s*([123])\\b",
            r"(?im)^\\s*([123])\\s*(?:because\\b|$)",
            r"(?i)\\(\\s*\\[[^\\]]+\\]\\s*:\\s*([123])\\b",
            r"(?i)\\bevaluation\\s*result\\b[^0-9]{0,20}([123])\\b",
            r"(?i)\\bscore\\b[^0-9]{0,20}([123])\\b",
            r"(?i)\\brating\\b[^0-9]{0,20}([123])\\b",
            r"(?i)\\boverall\\b[^0-9]{0,20}([123])\\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, candidate)
            if not match:
                continue
            value = match.group(match.lastindex or 0)
            try:
                numeric_value = int(value)
            except Exception:
                continue
            if 1 <= numeric_value <= 3:
                return numeric_value
        return None

    segments = []
    start_index = text.find("Updated Video Description")
    if start_index != -1:
        all_content = text[start_index + len("Updated Video Description"):].strip()
        eval_result_index = all_content.find("Evaluation Result")
        if eval_result_index != -1:
            remaining_content = all_content[eval_result_index + len("Evaluation Result"):].strip()
            because_index = remaining_content.lower().find("because")
            target = remaining_content if because_index == -1 else remaining_content[:because_index].strip()
            segments.append(target)
    segments.append(text)

    for segment in segments:
        parsed = parse_segment(segment)
        if parsed is not None:
            return parsed

    print("No information found.")
    return 0
"""
    if old_extract in text:
        text = text.replace(old_extract, new_extract, 1)
        changed = True

    path.write_text(text, encoding="utf-8")
    return changed


def patch_openai_eval_module(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "OPENAI_BASE_URL" in text and "GPT4o_MODEL" in text:
        return False

    pattern = (
        r"client = OpenAI\(\s*api_key\s*=\s*config\['GPT4o_API_KEY'\],\s*base_url\s*=\s*config\['GPT4o_BASE_URL'\]\s*\)\s*\n\s*MODEL\s*=\s*[\"']gpt-4o-2024-08-06[\"']"
    )
    replacement = """client = OpenAI(
        api_key=config.get('GPT4o_API_KEY', config.get('OPENAI_API_KEY', 'local-videobench')),
        base_url=config.get('GPT4o_BASE_URL', config.get('OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1'))
    )
    MODEL = config.get('GPT4o_MODEL', config.get('OPENAI_MODEL', 'gpt-4o-2024-08-06'))"""
    new_text, count = re.subn(pattern, replacement, text, flags=re.S)
    if count != 1:
        raise RuntimeError(f"Unable to patch OpenAI client config in {path}; count={count}")
    path.write_text(new_text, encoding="utf-8")
    return True


def patch_openai_eval_modules(root: Path) -> list[str]:
    changed = []
    for name in [
        'dynamicquality.py',
        'dynamicquality_gridview_customized.py',
        'staticquality.py',
        'staticquality_customized.py',
    ]:
        path = root / name
        if patch_openai_eval_module(path):
            changed.append(str(path))
    return changed


def patch_videobench_install() -> int:
    spec = importlib.util.find_spec("videobench")
    if spec is None or not spec.submodule_search_locations:
        print("videobench package not found")
        return 1
    root = Path(spec.submodule_search_locations[0])
    changed = []
    if patch_prompt_dict(root):
        changed.append(str(root / "prompt_dict.py"))
    if patch_init(root):
        changed.append(str(root / "__init__.py"))
    if patch_video_text_alignment(root):
        changed.append(str(root / "VideoTextAlignment.py"))
    changed.extend(patch_openai_eval_modules(root))
    for rel in [
        root / "prompt_dict.py",
        root / "__init__.py",
        root / "VideoTextAlignment.py",
        root / "dynamicquality.py",
        root / "dynamicquality_gridview_customized.py",
        root / "staticquality.py",
        root / "staticquality_customized.py",
    ]:
        py_compile.compile(str(rel), doraise=True)
    if changed:
        print("patched files:")
        for item in changed:
            print(item)
    else:
        print("videobench package already patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(patch_videobench_install())
