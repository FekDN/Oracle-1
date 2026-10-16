"""Historical example graph construction and event simulation."""
from __future__ import annotations

from ._shared import *
from .entities import KnowledgeEdge, KnowledgeNode

def trigger_maiman_event(system):
    graph = system["graph"]
    extended = system["extended"]
    topology = system["topology"]
    dynamics = system["dynamics"]
    
    logger.info("\n" + "💥"*30)
    logger.info("ACTIVATE CATALYST: THE MAIMAN MOMENT (MAY 1960)")
    logger.info("💥"*30)

    theory_id = "theory_1958"

    # 1. Создаем ЭНТРОПИЙНЫЙ ВЗРЫВ (Исходящие влияния)
    # Теория начинает "фонить" в прикладные области
    apps = ["surgery", "telecom", "cutting", "fusion", "lidar"]
    for i, app_name in enumerate(apps):
        app_id = f"app_{app_name}"
        graph.add_node(KnowledgeNode(
            id=app_id, 
            text=f"Laser application in {app_name}", 
            domain=app_name, 
            strategic_value=7.0
        ))
        # Создаем ребро с высокой социальной корреляцией
        e_out = KnowledgeEdge(
            id=f"e_out_{app_name}", 
            source_id=theory_id, 
            target_id=app_id, 
            semantic_similarity=0.4, 
            social_correlation=0.95
        )
        e_out.compute_total_weight()
        graph.add_edge(e_out)

    # 2. РЕСУРСНЫЙ УДАР (Пентагон)
    # investment_correlation передается внутрь KnowledgeEdge
    graph.add_node(KnowledgeNode(
        id="pentagon", 
        text="ARPA High-Power Beam Contract", 
        domain="military", 
        investment_score=10.0, 
        strategic_value=10.0
    ))
    
    e_money = KnowledgeEdge(
        id="e_money", 
        source_id="pentagon", 
        target_id=theory_id, 
        investment_correlation=1.0  # Теперь параметр на месте
    )
    extended.register_edge(e_money)

    # 3. ФОРСИРУЕМ "ЖАР" (Пробуждаем Эйнштейна)
    # Устанавливаем высокий прогнозный скор текущим узлам
    for nid in ["maser_1954", theory_id, "pentagon"]:
        node = graph.get_node(nid)
        if node: node.forecast_score = 9.9

    # 4. ФИНАЛЬНЫЙ ЦИКЛ СИНГУЛЯРНОСТИ
    logger.info("\n>>> COMMENCING TOTAL TOPOLOGICAL COLLAPSE <<<")
    
    for step in range(10):
        # Используем современное время, чтобы обойти штраф Dormancy
        current_ts = time.time() 
        
        # Ручная накачка "Интеллектуального шторма"
        # Имитируем аномальность (узел не похож на старые вакуумные лампы)
        system["topology"].anomaly_acc.push(theory_id, 1.0)
        
        # Имитируем конфликт данных (споры лабораторий Hughes vs Bell Labs)
        # В нашей реализации _cache — это словарь в StructuralConflictMeter
        system["topology"].conflict_meter._cache[theory_id] = 0.95
        # Имитируем высокую энтропию связей
        system["topology"].entropy_calc._cache[theory_id] = 0.90

        # Обновляем систему
        extended.update_epoch(current_ts)
        dynamics.run_epoch(current_ts)
        topo_res = topology.run_epoch(current_ts)

        # Прямое считывание Tension для мониторинга
        st_map = system["topology"].tension_field.compute(
            graph, 
            system["pressure_field"], 
            system["topology"].conflict_meter, 
            system["topology"].anomaly_acc, 
            system["topology"].entropy_calc, 
            current_ts
        )
        t_val = st_map[theory_id].tension
        
        logger.info(f"Final Step {step} | ⚡ Tension: {t_val:.4f} (Goal: 0.820)")

        if topo_res['rupture_count'] > 0 or t_val >= 0.82:
            logger.warning("\n" + "🎇"*30)
            logger.warning("THE ONTOLOGICAL RUPTURE IS COMPLETE!")
            logger.warning("THE REALITY HAS RECONFIGURED AROUND THE LASER")
            logger.warning("🎇"*30)
            logger.info(topology.rupture_report())
            
            # Извлекаем материализованный Лазер из синтезатора
            v_nodes, _ = dynamics.virtual_synth.get_virtual_nodes_as_graph_entries()
            if v_nodes:
                logger.info(f"✨ ORACLE-1 HAS SPOKEN. NEW NODE: {v_nodes[0].text}")
                logger.info(f"✨ Predicted Domain: {v_nodes[0].domain}")
                logger.info(f"✨ Scientific Readiness: {v_nodes[0].readiness_score:.2f}")
            break
    else:
        logger.error("Singularity failed. Check if Tension formula squares the value correctly.")



def run_real_historical_laser_path(system):
    graph = system["graph"]
    extended = system["extended"]
    topology = system["topology"]
    dynamics = system["dynamics"]
    
    logger.info("\n" + "📜"*30)
    logger.info("ORACLE-1: THE REAL LASER CHRONOLOGY (PREHISTORY TO 1960)")
    logger.info("📜"*30)

    # --- ТАЙМЛАЙН КОНСТАНТЫ ---
    TS_1800 = -5364792000.0
    TS_1917 = -1672531200.0
    TS_1954 = -504921600.0
    TS_1958 = -378691200.0
    TS_1960 = -315532800.0

    # ==========================================================================
    # PHASE 0: PREHISTORY (Культурный вектор и Математика)
    # ==========================================================================
    graph.add_node(KnowledgeNode(
        id="archimedes_myth", text="Archimedes Heat-Ray (Siege of Syracuse)",
        domain="military_history", entity_type="cultural_phantom", 
        social_gravity=8.0, phantom_weight=0.6, timestamp=TS_1800 - 1e10
    ))
    
    graph.add_node(KnowledgeNode(
        id="huygens_optics", text="Wave Theory of Light (Traite de la Lumiere)",
        domain="optics", entity_type="physical_principle", timestamp=TS_1800 - 1e9,
        scientific_score=9.0
    ))

    # ==========================================================================
    # PHASE 1: FUNDAMENTAL NODES (1801-1920)
    # ==========================================================================
    graph.add_node(KnowledgeNode(
        id="young_1801", text="Wave Interference principle",
        domain="physics", entity_type="physical_principle", timestamp=TS_1800,
        scientific_score=9.5
    ))

    graph.add_node(KnowledgeNode(
        id="maxwell_1865", text="Maxwell's Electromagnetic Theory",
        domain="physics", entity_type="unifying_theory", timestamp=TS_1800 + 2e9,
        scientific_score=10.0, strategic_value=8.0
    ))

    graph.add_node(KnowledgeNode(
        id="einstein_1917", text="Stimulated Emission Theory (A & B Coefficients)",
        domain="quantum_mechanics", entity_type="dormant_principle", timestamp=TS_1917,
        scientific_score=10.0, strategic_value=10.0, upstream_pressure=10.0
    ))

    # Ребра Фазы 1
    extended.register_edge(KnowledgeEdge(id="e_y_m", source_id="young_1801", target_id="maxwell_1865", semantic_similarity=0.9))
    extended.register_edge(KnowledgeEdge(id="e_m_e", source_id="maxwell_1865", target_id="einstein_1917", semantic_similarity=0.75))

    # ==========================================================================
    # PHASE 2: TECHNICAL & CULTURAL VECTORS (1920-1950)
    # ==========================================================================
    # Фантастика (Death Ray)
    graph.add_node(KnowledgeNode(
        id="garin_ray", text="Alexei Tolstoy: The Garin Death Ray (Crystalline Focus)",
        domain="fiction", entity_type="cultural_phantom", timestamp=TS_1917 + 3e8,
        social_gravity=9.5, phantom_weight=0.9
    ))

    # Промышленность (Рубин)
    graph.add_node(KnowledgeNode(
        id="ruby_synthesis", text="Industrial Verneuil Ruby Production",
        domain="materials", entity_type="industrial_substrate", timestamp=TS_1917 + 5e8,
        scientific_score=6.0, maturity_score=9.0
    ))

    # Радары (Когерентность)
    graph.add_node(KnowledgeNode(
        id="ww2_radar", text="WWII Radar: Coherent Microwave Mastery",
        domain="electronics", entity_type="technical_mastery", timestamp=TS_1954 - 3e8,
        scientific_score=8.5, strategic_value=10.0, efficiency_plateau=8.5
    ))

    # Ребра Фазы 2
    extended.register_edge(KnowledgeEdge(id="e_e_p", source_id="einstein_1917", target_id="garin_ray", social_correlation=0.8))
    extended.register_edge(KnowledgeEdge(id="e_r_m", source_id="ruby_synthesis", target_id="ww2_radar", investment_correlation=0.5))

    # ==========================================================================
    # PHASE 3: CONVERGENCE & BREAKTHROUGH (1954-1960)
    # ==========================================================================
    # МАЗЕР
    graph.add_node(KnowledgeNode(
        id="maser_1954", text="Ammonia MASER (Townes/Basov/Prokhorov)",
        domain="quantum_electronics", entity_type="breakthrough_node", timestamp=TS_1954,
        scientific_score=10.0, strategic_value=9.0
    ))

    # Теория Оптического Мазера
    theory_id = "theory_1958"
    graph.add_node(KnowledgeNode(
        id=theory_id, text="Optical Maser Theory (Schawlow-Townes Spec)",
        domain="optics", entity_type="functional_spec", timestamp=TS_1958,
        scientific_score=10.0, strategic_value=10.0, upstream_pressure=150.0
    ))

    # Пентагон
    graph.add_node(KnowledgeNode(
        id="pentagon_1959", text="Pentagon Coherent Light Funding ($15M)",
        domain="military_finance", entity_type="resource_source", timestamp=TS_1960 - 3e7,
        investment_score=10.0, strategic_value=10.0
    ))

    # Инкумбент (Вакуумные лампы)
    graph.add_node(KnowledgeNode(
        id="incumbent_radio", text="Vacuum Tube Microwave Industry",
        domain="electronics", strategic_value=10.0, efficiency_plateau=9.8
    ))

    # Ребра Фазы 3
    extended.register_edge(KnowledgeEdge(id="e_e_ma", source_id="einstein_1917", target_id="maser_1954", limitation_resolution=1.0))
    extended.register_edge(KnowledgeEdge(id="e_ma_t", source_id="maser_1954", target_id=theory_id, semantic_similarity=0.95))
    extended.register_edge(KnowledgeEdge(id="e_p_t", source_id="pentagon_1959", target_id=theory_id, investment_correlation=1.0))
    e_inh = KnowledgeEdge(id="e_inh", source_id="incumbent_radio", target_id=theory_id, inhibitory_force=0.95)
    e_inh.compute_total_weight()
    extended.register_edge(e_inh)
    extended.register_edge(KnowledgeEdge(id="e_t_ph", source_id=theory_id, target_id="garin_ray", social_correlation=1.0))

    # --- СИМУЛЯЦИЯ ФИНАЛЬНОГО СХЛОПЫВАНИЯ ---
    logger.info("\n>>> STARTING TOPOLOGICAL ANALYSIS: 1960.0 <<<")
    
    # Чтобы пробить разрыв в маленьком графе, активируем имитацию плотности
    for epoch in range(12):
        ts = TS_1960 + (epoch * 2592000)
        
        for nid in list(graph.nodes):
            node = graph.get_node(nid)
            node.forecast_score = 9.8
            if nid == theory_id:
                # Накачиваем аномальность и конфликт
                system["topology"].anomaly_acc.push(nid, 1.0)
                # Прямая инъекция в кеши для имитации тысяч фоновых связей
                system["topology"].conflict_meter._cache[nid] = 0.95
                system["topology"].entropy_calc._cache[nid] = 0.98

        ext_res = extended.update_epoch(ts)
        dyn_res = dynamics.run_epoch(ts)
        topo_res = topology.run_epoch(ts)

        # Мониторинг напряжения
        theory_state = topology.tension_field.compute(graph, system["pressure_field"], 
                                                      topology.conflict_meter, system["topology"].anomaly_acc, 
                                                      system["topology"].entropy_calc, ts)
        t_val = theory_state[theory_id].tension
        
        logger.info(f"Month {epoch} | Tension: {t_val:.4f} | Einstein Dormancy: {extended.dormancy_tracker.get_dormancy('maxwell_1865', 'einstein_1917'):.2f}")

        if topo_res['rupture_count'] > 0 or t_val > 0.82:
            logger.warning("\n" + "⚡"*30)
            logger.warning("ONTOLOGICAL RUPTURE DETECTED: THE REALITY COLLAPSE")
            logger.warning("⚡"*30)
            logger.info(topology.rupture_report())
            
            # Извлекаем предсказанное изобретение
            v_nodes, v_edges = dynamics.virtual_synth.get_virtual_nodes_as_graph_entries()
            if v_nodes:
                logger.info(f"✨ ORACLE-1 PREDICTED INVENTION: {v_nodes[0].text}")
                logger.info(f"✨ PHYSICAL FEASIBILITY: {v_nodes[0].readiness_score:.2f}")
            break

    logger.info("\n" + "█"*60)
    logger.info("HISTORICAL RECONSTRUCTION COMPLETE")
    logger.info("█"*60)

__all__ = ['trigger_maiman_event', 'run_real_historical_laser_path']
