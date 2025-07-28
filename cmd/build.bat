@echo off
del fmod_importer.zip 2>nul
powershell -Command "Compress-Archive -Path .\src\* -DestinationPath .\fmod_importer.zip -Force"