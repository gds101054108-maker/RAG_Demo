@echo off
echo ========================================
echo RAG 系统环境激活
echo ========================================
echo.

call venv\Scripts\activate.bat

echo 虚拟环境已激活！
echo.
echo 可用命令:
echo   python main.py ingest --dir ./data/documents  # 摄取文档
echo   python main.py query "问题"                    # 查询
echo   python main.py interactive                    # 交互模式
echo.

cmd /k