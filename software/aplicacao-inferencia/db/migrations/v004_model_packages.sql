-- v004_model_packages: adiciona suporte à importação de pasta completa de pacote RecycleAI.
--
-- Backward compatible: package_dir é NULL para modelos legados (importados como arquivo único).
-- Para modelos importados como pacote completo, package_dir contém o caminho relativo da
-- pasta <nome>_package/ à raiz do projeto (posix), e file_path aponta diretamente para
-- o arquivo de deploy declarado no manifest (ex: runtime_inferencia/modelos_importados/<pkg>/weights/best_ts.pt).

ALTER TABLE models ADD COLUMN package_dir TEXT;
-- NULL  → modelo importado como arquivo único (legado)
-- TEXT  → caminho posix relativo à raiz do projeto, ex: "runtime_inferencia/modelos_importados/recycleai_pkg"
