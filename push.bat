@echo off
echo ==============================================
echo Pushing Zoya AI Assistant to GitHub...
echo ==============================================

git add .
git commit -m "feat: initial commit with core, automation and Gemini LLM modules"
git branch -M main
git remote add origin https://github.com/Nisarg266/Zoya-Ai-assistant-.git 2>nul
git push -u origin main

echo.
echo ==============================================
echo Done! Please check your GitHub repository.
echo ==============================================
pause
