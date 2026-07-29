

  /home/kuceral4/projects/motion_toolkit  ->  ~/.claude/projects/-home-kuceral4-projects-motion-toolkit
  /home/kuceral4/projects/helhest_stack   ->  ~/.claude/projects/-home-kuceral4-projects-helhest-stack

  How to move it — one mv of the whole slug dir carries both memory and history:

  mv ~/.claude/projects/-home-kuceral4-projects-motion-toolkit \
     ~/.claude/projects/-home-kuceral4-projects-helhest-stack

  Important — do it when this session is closed. That directory contains this session's live transcript (1748341b-….jsonl), so moving it out from under a running session is asking for trouble. So the full sequence, after you exit Claude:

  # 1. rename the repo/dir (from earlier)
  gh repo rename helhest_stack
  mv ~/projects/motion_toolkit ~/projects/helhest_stack
  git -C ~/projects/helhest_stack remote set-url origin git@github.com:aleskucera/helhest_stack.git

  # 2. move the Claude project dir (memory + history) to match the new path
  mv ~/.claude/projects/-home-kuceral4-projects-motion-toolkit \
     ~/.claude/projects/-home-kuceral4-projects-helhest-stack

  # 3. relaunch from the new location
  cd ~/projects/helhest_stack && claude

  Claude will then find your memory and past sessions under the new slug automatically.
