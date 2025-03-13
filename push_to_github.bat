@echo off
echo Pushing changes to GitHub...

git add PROJECT_REPORT.md README.md docs/images
git commit -m "Add detailed project report and documentation"
git push

echo Done! 