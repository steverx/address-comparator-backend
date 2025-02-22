import { pool } from '../config/database.config';

async function initializeDatabase() {
    try {
        console.log('Creating addresses table...');
        await pool.query(`
            CREATE TABLE IF NOT EXISTS addresses (
                id SERIAL PRIMARY KEY,
                raw_address TEXT NOT NULL,
                normalized_address TEXT NOT NULL,
                components JSONB DEFAULT '{}',
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_normalized_address 
            ON addresses(normalized_address);
        `);
        console.log('Database initialization completed successfully');
    } catch (error) {
        console.error('Database initialization failed:', error);
        throw error;
    } finally {
        await pool.end();
    }
}

initializeDatabase();