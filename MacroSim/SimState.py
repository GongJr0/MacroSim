from dataclasses import dataclass


@dataclass
class SimState:
    # Demographic state
    population: int  # Count
    literacy: float  # % population
    labor_force: float  # % population
    net_migration: int  # Count
    birth_rate: float  # Births per 1,000 people
    death_rate: float  # Deaths per 1,000 people
    dependency_ratio: float  # Non-working population / labor force

    # Technological state
    domestic_tech: float  # % total tech
    foreign_dependent_tech: float  # % total tech
    rd_investment: float  # % of GDP spent on R&D

    # Political state
    corruption: float  # % perception index
    support_lvl: float  # % political support
    rule_of_law: float  # 0-100 scale
    political_stability: float  # -2.5 to 2.5 (World Bank scale)

    # Economic state
    gdp_nom: float
    gdp_real: float
    consumption: float
    real_wage: float
    savings: float
    investment: float
    gov_exp: float
    avg_tax: float
    money_supply: float
    depreciation: float
    fdi: float
    inflation: float
    expected_inflation: float
    fed_interest: float
    nx: float  # Net exports
    labor_productivity: float  # Output per worker
    trade_balance: float  # Exports - Imports
    debt_gdp_ratio: float  # % of GDP
    unemployment: float  # % of labor force

    # Energy & Resources
    energy_prod: float  # Total energy production
    energy_dep: float  # % energy imported
    resource_exports: float  # % of GDP from resource exports
